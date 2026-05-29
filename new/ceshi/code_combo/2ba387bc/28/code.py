# Combo variant of 2ba387bc (index 28)
# combo_spec: D2/2+D10/0
# Name: 单列输出+距离排序

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location

"""
Design analysis for combo D2/2 + D10/0

a) 具体化与行为追踪
   - 原题输入仍是干净背景上的若干 4x4 方块，两类对象结构不变：
     1. frame：单色外框，中心 2x2 是背景。
     2. solid：4x4 全填充。
   - 原题输出把 frame 堆到左列、solid 堆到右列，并且每组内部按 (row, col) 排序。
   - 本融合变体同时改两处，而且两处必须连成一条推理链：
     1. 不再分类。frame 与 solid 全部放进同一个序列。
     2. 这个统一序列按“方块中心到整张网格中心的欧氏距离”从近到远排序。
     3. 输出只保留一个宽 4 的列，按该统一顺序自上而下堆叠。
   - 具体例子：假设输入里有 4 个对象，按输入位置顺序分别是
     F@(1,1), S@(14,14), F@(9,5), S@(18,3)，网格大小 24x24。
     原题会产生两列：左列 [F@(1,1), F@(9,5)]，右列 [S@(14,14), S@(18,3)]。
     本变体先算四个对象到中心 (11.5,11.5) 的距离，可能得到顺序
     [F@(9,5), S@(14,14), S@(18,3), F@(1,1)]，然后把这四个对象全部按这个次序
     堆成单列。差异不只是“排序键变了”，而是“排序域”也从组内变成全体对象。

b) 复杂度校准
   - 没有新增对象种类或隐藏标记，仍只有 frame / solid 两种显式可见形态。
   - 复杂度主要来自两个可见变化：
     1. 输出从双列变成单列。
     2. 单列内 frame 和 solid 会混合，且顺序由与中心距离决定，而不是由行位置决定。
   - 为避免过度简化，generator 要求样本里两类对象都出现，并且距离排序后的类型序列不能退化成
     “所有 frame 在前 / 所有 solid 在前”的单块结构，否则“不分类”的证据太弱。
   - 为避免过度复杂，仍只处理 4x4 方块，不引入额外颜色规则或旋转规则。

c) 隐式假设审查
   - `find_components(... monochromatic=False)` 假设对象彼此不接触，否则会合并组件。
     该前提仍成立，generator 继续用 `padding=1` 保证分离。
   - `bbox_h == bbox_w == 4` 假设对象都是 4x4；融合变体不改形状，因此继续成立。
   - frame 判定逻辑依赖 “边框单色且中心 2x2 为背景”；该结构不变，继续成立。
   - 原题的 `frames.sort(...)` / `solids.sort(...)` 依赖“先分类、再组内排序”的前提。
     D10/0 使这个前提失效，所以 solver 必须改成“统一收集后只排序一次”。
   - D2/2 引入新的假设：距离最好唯一，否则主规则不明显。solver 用 `(dist2, row, col)` 做稳定
     决定性 tie-break；generator 额外避免等距样本，让人类真正能看出“距离排序”。
   - 新冲突检查：如果仍沿用原来的输出高度 `4 * max(#frames, #solids)` 会在单列模式下丢对象；
     因此必须改为 `4 * total_objects`。这是两个维度叠加后才出现的直接结构性影响。

d) 放置合法性
   - generator 只在足够大的空白网格上放置 4x4 patch，并通过 `random_free_location(... border=1,
     padding=1, padding_connectivity=4)` 保证不重叠、不越界、不共享边。
   - 每个对象使用唯一非背景颜色，避免组件提取或视觉识别歧义。
   - 生成后调用 `main()` 做同步验证，确保 solver 可处理。

e) 视觉可观察性
   - 排序依据是“谁更靠近整图中心”，这是纯空间关系，人类可直接看见。
   - 不分类也必须是肉眼可见的，所以输出不只变窄，还要求单列中的类型顺序出现交错：
     距离序列里既有 frame 在 solid 前，也有 solid 在 frame 前。
   - generator 同时要求距离排序与旧的 `(row, col)` 排序不同，否则即便单列成立，D2 变化仍可能不明显。

f) 可视化整洁度
   - 输入仍是分离的 4x4 小方块，背景干净，不做覆盖叠加。
   - 输出只是一条 4 列宽的竖直堆栈，视觉上比双列更简单，不会引入新的拥挤区域。

g) 变化的可推断性
   - 不涉及颜色映射变化；颜色只用于区分对象身份。
   - 颜色种类最多 8 个，且每个对象颜色唯一，便于把输入对象与输出条目一一对应，再观察它们是按
     “离中心远近”而非“形状类别 / 行位置”排列。

h) 新增假设与 generator 保证
   - `main()` 的新增关键假设：
     1. 至少有一个 4x4 对象。
     2. 若存在等距对象，用 `(row, col)` 打破平局即可。
   - `generate_input()` 进一步保证强可观察性：
     1. frame 和 solid 都至少各有 2 个；
     2. 所有对象的 `dist2` 两两不同；
     3. 全体对象的距离排序 != 原始的 `(row, col)` 排序；
     4. 距离排序后的类型序列同时包含 “frame 在 solid 前” 和 “solid 在 frame 前”，
        从而真正体现“混合而非分组”。
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
    Squared Euclidean distance using doubled coordinates to keep everything integral.

    Object center: (r0 + 1.5, c0 + 1.5) -> (2*r0 + 3, 2*c0 + 3)
    Grid center: ((H - 1)/2, (W - 1)/2) -> (H - 1, W - 1)
    """
    dr2 = int(2 * int(r0) + 3 - (int(H) - 1))
    dc2 = int(2 * int(c0) + 3 - (int(W) - 1))
    return int(dr2 * dr2 + dc2 * dc2)


def _collect_objects(g: np.ndarray) -> tuple[int, list[dict]]:
    bg = int(get_bg(g))
    H, W = map(int, g.shape)
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")

    objects: list[dict] = []
    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 4 or bbox_w(b) != 4:
            continue
        r0, c0, _r1, _c1 = map(int, b)
        sub = crop_bbox(g, b)
        kind = "frame" if _is_frame(sub, bg) else "solid"
        objects.append(
            {
                "r0": int(r0),
                "c0": int(c0),
                "sub": sub.copy(),
                "kind": kind,
                "dist2": _distance2_to_center(int(r0), int(c0), int(H), int(W)),
            }
        )

    return int(bg), objects


def _distance_sorted(objects: list[dict]) -> list[dict]:
    return sorted(objects, key=lambda obj: (int(obj["dist2"]), int(obj["r0"]), int(obj["c0"])))


def _position_order_signature(objects: list[dict]) -> tuple[tuple[int, int], ...]:
    ordered = sorted(objects, key=lambda obj: (int(obj["r0"]), int(obj["c0"])))
    return tuple((int(obj["r0"]), int(obj["c0"])) for obj in ordered)


def _distance_order_signature(objects: list[dict]) -> tuple[tuple[int, int], ...]:
    ordered = _distance_sorted(objects)
    return tuple((int(obj["r0"]), int(obj["c0"])) for obj in ordered)


def _type_sequence(objects: list[dict]) -> tuple[str, ...]:
    return tuple(str(obj["kind"]) for obj in objects)


def _behavior_metrics(g: np.ndarray) -> dict[str, object]:
    """
    Quantify whether the generated sample really exhibits the combo behavior.

    Metrics:
    - `unique_distances`: all objects have distinct center distances.
    - `reordered_vs_row`: distance order differs from original (row, col) order.
    - `mixed_types_in_distance_order`: the unified sorted sequence is not separable into
      one frame block plus one solid block.
    - `type_switches`: number of adjacent type changes in the sorted sequence.
    """
    _bg, objects = _collect_objects(np.array(g, dtype=int))
    ordered = _distance_sorted(objects)
    seq = _type_sequence(ordered)

    has_frame_before_solid = False
    has_solid_before_frame = False
    for i, left in enumerate(seq):
        for right in seq[i + 1 :]:
            if left == "frame" and right == "solid":
                has_frame_before_solid = True
            if left == "solid" and right == "frame":
                has_solid_before_frame = True

    return {
        "n_objects": int(len(objects)),
        "n_frames": int(sum(obj["kind"] == "frame" for obj in objects)),
        "n_solids": int(sum(obj["kind"] == "solid" for obj in objects)),
        "unique_distances": len({int(obj["dist2"]) for obj in objects}) == len(objects),
        "reordered_vs_row": _position_order_signature(objects) != _distance_order_signature(objects),
        "type_switches": int(sum(seq[i] != seq[i - 1] for i in range(1, len(seq)))),
        "mixed_types_in_distance_order": bool(has_frame_before_solid and has_solid_before_frame),
        "distance_type_sequence": seq,
    }


def _variant_visible(g: np.ndarray) -> bool:
    metrics = _behavior_metrics(g)
    return bool(
        metrics["n_frames"] >= 2
        and metrics["n_solids"] >= 2
        and metrics["unique_distances"]
        and metrics["reordered_vs_row"]
        and metrics["mixed_types_in_distance_order"]
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stack all scattered 4x4 objects into a single 4-wide column.

    Combo rule:
    - Detect every 4x4 object exactly as in the original task.
    - Do not separate frame and solid objects.
    - Sort the full object set by distance from object center to grid center.
    - Paste them top-to-bottom into one column of width 4.
    """
    g = np.array(grid, dtype=int)
    bg, objects = _collect_objects(g)
    assert objects

    ordered = _distance_sorted(objects)
    outH = 4 * int(len(ordered))
    outW = 4
    out = np.full((outH, outW), int(bg), dtype=int)

    for i, obj in enumerate(ordered):
        paste(out, obj["sub"], int(i * 4), 0)

    return out


def _sample_unique_color(used_cols: set[int]) -> int:
    col = int(np.random.randint(1, 10))
    while col in used_cols:
        col = int(np.random.randint(1, 10))
    used_cols.add(col)
    return int(col)


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    H = int(np.random.randint(24, 33))
    W = int(np.random.randint(24, 33))
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
    for _ in range(800):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 800 attempts")

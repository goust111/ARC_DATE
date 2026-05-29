# Combo variant of 2ba387bc (index 50)
# combo_spec: D1/2+D2/2+D3/0+D11/0
# Name: 面积分类+距离排序+可变尺寸+干扰物

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

for _parent in Path(__file__).resolve().parents:
    _cand = _parent / "solvers" / "arc_utils.py"
    if _cand.exists():
        _solvers_dir = _cand.parent
        if str(_solvers_dir) not in sys.path:
            sys.path.append(str(_solvers_dir))
        break

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location


"""
Design analysis for the combo variant
=====================================

a) 具体化与行为追踪
   - 原题输入只含彼此分离的 4x4 方块；框架靠“2x2 内部是背景”识别，实心则是剩余的 4x4；
     每组再按 (row, col) 排序，输出固定为 8 列、每带高 4。
   - 融合后，真正有效对象改为 3x3 到 5x5 的单色正方形，分两类：
     1. frame：1 像素厚边框，内部全是背景，因此非背景面积 = 4*n-4。
     2. solid：整块填满，因此非背景面积 = n*n。
   - 输入里还会混入少量非标准矩形干扰物，例如 2x3、1x6、3x6；其中至少一个会贴着有效方块。
     如果继续用 multicolor component，贴边干扰会把有效方块并坏，所以 solver 必须先识别背景，
     再改用 monochromatic component，把每种颜色各自单独取出。
   - 小追踪例子：
     * 输入里有一个 5x5 框架、一个 3x3 实心、一个 4x4 实心，以及一个贴在 5x5 框架右边的 2x3 干扰条。
     * 原逻辑会在 `find_components(... monochromatic=False)` 时把 5x5 框架和 2x3 干扰并成一个多色大块，
       然后又因为 bbox 不是 4x4 直接丢失；剩余对象还会错误地按输入行序堆叠。
     * 融合逻辑则会：
       1. 找背景色；
       2. 做单色连通分量，贴边干扰被单独分离；
       3. 对每个 3..5 正方形看面积：5x5 框架面积 16 = 4*5-4，所以归 frame；3x3 实心面积 9、
          4x4 实心面积 16 = n*n，所以归 solid；
       4. 再比较每个方块中心到整张网格中心的欧氏距离，离中心近的先出现在各自列的上方。
   - 差异不是单点替换，而是完整四步链条：背景识别 -> 干扰过滤 -> 面积分组 -> 距离排序。

b) 复杂度校准
   - 有效尺寸只允许 n in {3,4,5}，避免自由度过多；干扰物只用一小组简单矩形，视觉上容易看出是“非标块”。
   - 面积分类并非纯数字谜题：n 变化很小，观察者能直接看到 3x3/4x4/5x5 的边框像素数与实心像素数不同。
   - generator 强制：
     * 两组对象都至少 2 个，保证排序规律有重复支撑；
     * 至少一个 frame 和至少一个 solid 不是 4x4，避免退化到原题固定面积 12/16；
     * 至少一个干扰物贴边；
     * 至少一组的距离排序与旧的行序不同，避免样例表面上看起来还是原题。

c) 隐式假设审查（逐段对原 main 的相关逻辑）
   - `find_components(... monochromatic=False)`：
     原假设是“有效对象不会和异色块接触”。D11 破坏了它，所以必须改成 monochromatic=True。
   - `if bbox_h(b) != 4 or bbox_w(b) != 4`：
     原假设是固定 4x4。D3 破坏它，必须改成接受 3..5 的正方形。
   - `sub[1:3,1:3] == bg`：
     原假设是内部永远是 2x2。D3 破坏它，内部应改成 `sub[1:-1,1:-1]`。
   - `if frame else solid`：
     原逻辑默认“任何非 frame 的 4x4 都是 solid”。融合后不成立，必须显式检查面积与形状，
     否则某些正方形噪声会被误收。
   - `sort(key=(row,col))`：
     原假设是组内规则只看输入位置。D2 破坏它，必须改成看对象中心到网格中心的距离。
   - `outH = 4 * max(...)`, `outW = 8`, `paste(..., i*4, ...)`：
     原假设是对象尺寸固定。D3 破坏它，输出列宽必须等于全局最大尺寸，行高必须按每个配对 band 的较大尺寸累计。
   - 新增假设：
     * 有效对象都是单色、孤立的 3..5 正方形；
     * frame 的非背景面积恰好是 4*n-4，solid 的面积恰好是 n*n；
     * 同组距离最好不相等，这样“按距离排序”才对人类可见。
     generator 会逐条保证这些条件。

d) 放置合法性
   - generator 维护两张网格：
     * `g`：真实输入；
     * `occ`：保留占位网格。对 frame 在 `occ` 里也把内部孔洞视为已占用，防止后续干扰物落进孔里。
   - 有效对象先放，使用较大的 padding，保证它们彼此不重叠、不越界，并给贴边干扰留出空间。
   - 干扰物放置时要么自由散落，要么正交贴到某个目标 bbox 上；贴边版本必须：
     * 自身不与任何已占用区域重叠；
     * 至少接触一个目标对象；
     * 不能同时接触别的对象。

e) 视觉可观察性
   - 人类能直接看到三件事：对象尺寸在 3/4/5 间变化、输出里只保留规则正方形、并且离中心近的排在更上面。
   - 干扰物不参与输出，因此多组样例会反复展示“不是所有彩色块都算对象”。
   - 距离规则只使用几何中心，不依赖颜色编号、生成顺序或其它人类不可见因素。

f) 可视化整洁度
   - 每个有效对象使用唯一颜色，干扰物也尽量使用不同颜色，便于肉眼分开组件。
   - 有效对象之间留较大间距，避免多个贴边干扰在同一区域纠缠。
   - 输出仍保持双列堆叠的原始版式，只把列宽和 band 高度扩展到适配可变尺寸。

g) 变化的可推断性
   - 本融合不做颜色映射；颜色只区分实例。
   - 面积分类的“类别线索”来自几何结构本身：同样是正方形，空心边框和实心块在可见像素数上不同，
     且这种差异会随尺寸 3/4/5 一起重复出现。
   - 因为 5x5 frame 的面积 16 会和 4x4 solid 同数值，solver 不能只看裸面积，必须结合 bbox 尺寸；
     这仍然是“按面积分类”，只是面积阈值取决于可见尺寸 n。

h) 量化退化检查思路
   - 仅验证 `main(generate_input())` 不够，因为样本可能偶然按行序和按距离序相同。
   - 因此 generator 额外拒绝：
     * 所有对象都为 4x4；
     * 没有贴边干扰；
     * 距离排序输出与旧行序输出完全相同。
   - 统计指标使用 `_behavior_metrics()`：
     `frame_non4_count`, `solid_non4_count`, `attached_distractor_count`,
     `reordered_groups`, `distance_output_diff_from_row_output`。

i) main() 的新隐式假设与 generator 保证
   - `_classify_square_component()` 假设候选分量是单色的；generator 为每个对象和干扰物分配独立颜色。
   - `_distance2_to_center()` 假设需要 deterministic tie-break；main 用 `(dist2, row, col)`，
     generator 则要求每组距离平方互异。
   - `_assemble_output()` 假设至少存在一个有效对象；generator 永远放置 frame/solid 两组各至少两个。
   - 贴边干扰不会进入 frame 内孔，也不会贴住两个对象；`occ` 与 `_can_place_attached()` 共同保证。
"""


_DISTRACTOR_SHAPES: tuple[tuple[int, int], ...] = (
    (1, 5),
    (5, 1),
    (1, 6),
    (6, 1),
    (2, 3),
    (3, 2),
    (2, 4),
    (4, 2),
    (2, 5),
    (5, 2),
    (3, 6),
    (6, 3),
)


def _make_frame(size: int, col: int, bg: int = 0) -> np.ndarray:
    patch = np.full((int(size), int(size)), int(col), dtype=int)
    patch[1:-1, 1:-1] = int(bg)
    return patch


def _make_solid(size: int, col: int) -> np.ndarray:
    return np.full((int(size), int(size)), int(col), dtype=int)


def _distance2_to_center(r0: int, c0: int, size: int, H: int, W: int) -> int:
    dr2 = int(2 * int(r0) + int(size) - 1 - (int(H) - 1))
    dc2 = int(2 * int(c0) + int(size) - 1 - (int(W) - 1))
    return int(dr2 * dr2 + dc2 * dc2)


def _classify_square_component(sub: np.ndarray, color: int, area: int, bg: int) -> str | None:
    h, w = map(int, sub.shape)
    if h != w or h < 3 or h > 5:
        return None

    size = int(h)
    color = int(color)
    area = int(area)
    bg = int(bg)

    border = np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])
    border_colors = set(map(int, border.tolist()))
    full_area = int(size * size)
    frame_area = int(4 * size - 4)

    if area == full_area and bool((sub == color).all()):
        return "solid"

    if (
        area == frame_area
        and len(border_colors) == 1
        and color in border_colors
        and color != bg
        and bool((sub[1:-1, 1:-1] == bg).all())
    ):
        return "frame"

    return None


def _collect_components(g: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    bg = int(get_bg(g))
    H, W = map(int, g.shape)
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")

    objects: list[dict] = []
    distractors: list[dict] = []
    for comp in comps:
        b = comp["bbox"]
        h = int(bbox_h(b))
        w = int(bbox_w(b))
        r0, c0, _r1, _c1 = map(int, b)
        sub = crop_bbox(g, b)
        kind = _classify_square_component(sub, int(comp["color"]), int(comp["area"]), bg)
        if kind is None:
            distractors.append(
                {
                    "bbox": tuple(map(int, b)),
                    "color": int(comp["color"]),
                    "area": int(comp["area"]),
                    "coords": comp["coords"],
                    "shape": (int(h), int(w)),
                    "sprite": sub.copy(),
                }
            )
            continue
        size = int(h)
        objects.append(
            {
                "kind": kind,
                "size": size,
                "color": int(comp["color"]),
                "area": int(comp["area"]),
                "bbox": tuple(map(int, b)),
                "r0": int(r0),
                "c0": int(c0),
                "dist2": _distance2_to_center(int(r0), int(c0), size, int(H), int(W)),
                "sprite": sub.copy(),
            }
        )

    return int(bg), objects, distractors


def _split_groups(objects: list[dict]) -> tuple[list[dict], list[dict]]:
    frames = [obj for obj in objects if obj["kind"] == "frame"]
    solids = [obj for obj in objects if obj["kind"] == "solid"]
    return frames, solids


def _row_sorted(group: list[dict]) -> list[dict]:
    return sorted(group, key=lambda obj: (int(obj["r0"]), int(obj["c0"])))


def _distance_sorted(group: list[dict]) -> list[dict]:
    return sorted(group, key=lambda obj: (int(obj["dist2"]), int(obj["r0"]), int(obj["c0"])))


def _assemble_output(frames: list[dict], solids: list[dict], bg: int) -> np.ndarray:
    all_objs = list(frames) + list(solids)
    if not all_objs:
        return np.zeros((0, 0), dtype=int)

    max_size = max(int(obj["size"]) for obj in all_objs)
    row_count = max(int(len(frames)), int(len(solids)))
    row_heights = [
        max(
            int(frames[i]["size"]) if i < len(frames) else 0,
            int(solids[i]["size"]) if i < len(solids) else 0,
        )
        for i in range(row_count)
    ]

    out = np.full((int(sum(row_heights)), int(2 * max_size)), int(bg), dtype=int)
    out_r = 0
    for i, band_h in enumerate(row_heights):
        if i < len(frames):
            paste(out, frames[i]["sprite"], int(out_r), 0)
        if i < len(solids):
            paste(out, solids[i]["sprite"], int(out_r), int(max_size))
        out_r += int(band_h)
    return out


def _bbox_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, bbox)
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _component_touches_bbox(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    target = _bbox_cells(bbox)
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if (rr - 1, cc) in target or (rr + 1, cc) in target or (rr, cc - 1) in target or (rr, cc + 1) in target:
            return True
    return False


def _behavior_metrics(g: np.ndarray) -> dict:
    bg, objects, distractors = _collect_components(np.array(g, dtype=int))
    frames, solids = _split_groups(objects)
    row_frames = _row_sorted(frames)
    row_solids = _row_sorted(solids)
    dist_frames = _distance_sorted(frames)
    dist_solids = _distance_sorted(solids)

    row_out = _assemble_output(row_frames, row_solids, bg)
    dist_out = _assemble_output(dist_frames, dist_solids, bg)

    groups = [frames, solids]
    attached = 0
    for d in distractors:
        touches = sum(1 for obj in objects if _component_touches_bbox(d["coords"], obj["bbox"]))
        if touches >= 1:
            attached += 1

    return {
        "object_count": int(len(objects)),
        "frame_count": int(len(frames)),
        "solid_count": int(len(solids)),
        "distractor_count": int(len(distractors)),
        "attached_distractor_count": int(attached),
        "sizes": tuple(sorted({int(obj["size"]) for obj in objects})),
        "non4_count": int(sum(int(obj["size"]) != 4 for obj in objects)),
        "frame_non4_count": int(sum(int(obj["size"]) != 4 for obj in frames)),
        "solid_non4_count": int(sum(int(obj["size"]) != 4 for obj in solids)),
        "distance_unique_groups": int(
            sum(len({int(obj["dist2"]) for obj in group}) == len(group) for group in groups if group)
        ),
        "reordered_groups": int(
            sum(
                tuple((int(obj["r0"]), int(obj["c0"])) for obj in _row_sorted(group))
                != tuple((int(obj["r0"]), int(obj["c0"])) for obj in _distance_sorted(group))
                for group in groups
                if len(group) >= 2
            )
        ),
        "distance_output_diff_from_row_output": bool(not np.array_equal(row_out, dist_out)),
    }


def _variant_visible(g: np.ndarray) -> bool:
    metrics = _behavior_metrics(g)
    return bool(
        metrics["frame_count"] >= 2
        and metrics["solid_count"] >= 2
        and metrics["frame_non4_count"] >= 1
        and metrics["solid_non4_count"] >= 1
        and len(metrics["sizes"]) >= 2
        and metrics["attached_distractor_count"] >= 1
        and metrics["distance_unique_groups"] == 2
        and metrics["reordered_groups"] >= 1
        and metrics["distance_output_diff_from_row_output"]
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only valid 3x3..5x5 squares, classify them by area, and stack by center distance.

    Rule:
    - Detect background as the most common color.
    - Extract monochromatic 4-connected components so attached distractors do not merge with targets.
    - Keep only 3..5 square components that match one of two area-based templates:
      * frame: area = 4*n-4 with a bg-filled interior,
      * solid: area = n*n filled by one non-bg color.
    - Sort frames and solids independently by distance from object center to grid center.
    - Output uses two columns of width = max valid size; row bands are paired by index and take the larger height.
    """
    g = np.array(grid, dtype=int)
    bg, objects, _distractors = _collect_components(g)
    frames, solids = _split_groups(objects)
    frames = _distance_sorted(frames)
    solids = _distance_sorted(solids)
    return _assemble_output(frames, solids, bg)


def _sample_sizes(n_frames: int, n_solids: int) -> tuple[list[int], list[int]]:
    while True:
        frame_sizes = [int(np.random.randint(3, 6)) for _ in range(int(n_frames))]
        solid_sizes = [int(np.random.randint(3, 6)) for _ in range(int(n_solids))]
        all_sizes = frame_sizes + solid_sizes
        if len(set(all_sizes)) < 2:
            continue
        if not any(size != 4 for size in frame_sizes):
            continue
        if not any(size != 4 for size in solid_sizes):
            continue
        return frame_sizes, solid_sizes


def _can_place_attached(
    occ: np.ndarray,
    patch: np.ndarray,
    r0: int,
    c0: int,
    target_bbox: tuple[int, int, int, int],
    bg: int,
) -> bool:
    h, w = map(int, patch.shape)
    H, W = occ.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    if bool((occ[r0 : r0 + h, c0 : c0 + w] != int(bg)).any()):
        return False

    target_cells = _bbox_cells(target_bbox)
    touched_target = False
    touched_other = False
    for rr in range(int(r0), int(r0 + h)):
        for cc in range(int(c0), int(c0 + w)):
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if int(occ[nr, nc]) == int(bg):
                    continue
                if (int(nr), int(nc)) in target_cells:
                    touched_target = True
                else:
                    touched_other = True
    return bool(touched_target and not touched_other)


def _attached_location(
    occ: np.ndarray,
    patch: np.ndarray,
    target_bbox: tuple[int, int, int, int],
    bg: int,
) -> tuple[int, int] | None:
    r0, c0, r1, c1 = map(int, target_bbox)
    h, w = map(int, patch.shape)
    candidates: list[tuple[int, int]] = []

    if c0 - w >= 0:
        for rr in range(int(r0 - h + 1), int(r1) + 1):
            candidates.append((int(rr), int(c0 - w)))
    if c1 + 1 + w <= occ.shape[1]:
        for rr in range(int(r0 - h + 1), int(r1) + 1):
            candidates.append((int(rr), int(c1 + 1)))
    if r0 - h >= 0:
        for cc in range(int(c0 - w + 1), int(c1) + 1):
            candidates.append((int(r0 - h), int(cc)))
    if r1 + 1 + h <= occ.shape[0]:
        for cc in range(int(c0 - w + 1), int(c1) + 1):
            candidates.append((int(r1 + 1), int(cc)))

    if not candidates:
        return None
    order = np.random.permutation(len(candidates))
    for idx in order:
        rr, cc = candidates[int(idx)]
        if _can_place_attached(occ, patch, int(rr), int(cc), target_bbox, bg):
            return int(rr), int(cc)
    return None


def _place_distractor(
    g: np.ndarray,
    occ: np.ndarray,
    patch: np.ndarray,
    bg: int,
    anchor_bboxes: list[tuple[int, int, int, int]],
    force_attach: bool,
) -> tuple[bool, bool]:
    anchor_order = list(np.random.permutation(len(anchor_bboxes)).astype(int))

    if force_attach or bool(np.random.rand() < 0.65):
        for idx in anchor_order:
            loc = _attached_location(occ, patch, anchor_bboxes[int(idx)], bg)
            if loc is None:
                continue
            paste(g, patch, int(loc[0]), int(loc[1]), transparent=bg)
            paste(occ, patch, int(loc[0]), int(loc[1]), transparent=bg)
            return True, True
        if force_attach:
            return False, False

    try:
        r0, c0 = random_free_location(
            occ,
            patch,
            transparent=bg,
            border=1,
            padding=1,
            padding_connectivity=4,
        )
    except Exception:
        return False, False

    paste(g, patch, int(r0), int(c0), transparent=bg)
    paste(occ, patch, int(r0), int(c0), transparent=bg)
    return True, False


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    H = int(np.random.randint(30, 41))
    W = int(np.random.randint(30, 41))
    g = np.full((H, W), int(bg), dtype=int)
    occ = np.full((H, W), int(bg), dtype=int)

    n_frames = int(np.random.randint(2, 4))
    n_solids = int(np.random.randint(2, 4))
    n_valid = int(n_frames + n_solids)
    max_distractors = min(3, 9 - n_valid)
    n_distractors = int(np.random.randint(1, max_distractors + 1))

    colors = list(np.random.permutation(np.arange(1, 10)).astype(int))
    valid_colors = colors[:n_valid]
    distractor_colors = colors[n_valid : n_valid + n_distractors]
    frame_sizes, solid_sizes = _sample_sizes(n_frames, n_solids)

    valid_specs: list[dict] = []
    for i in range(n_frames):
        valid_specs.append({"kind": "frame", "size": int(frame_sizes[i]), "color": int(valid_colors[i])})
    for i in range(n_solids):
        valid_specs.append(
            {"kind": "solid", "size": int(solid_sizes[i]), "color": int(valid_colors[n_frames + i])}
        )

    order = np.random.permutation(len(valid_specs))
    objects: list[dict] = []
    anchor_bboxes: list[tuple[int, int, int, int]] = []

    try:
        for idx in order:
            spec = valid_specs[int(idx)]
            size = int(spec["size"])
            color = int(spec["color"])
            sprite = _make_frame(size, color, bg) if spec["kind"] == "frame" else _make_solid(size, color)
            reserve = np.full((size, size), int(color), dtype=int)
            r0, c0 = random_free_location(
                occ,
                reserve,
                transparent=bg,
                border=2,
                padding=3,
                padding_connectivity=4,
            )
            paste(g, sprite, int(r0), int(c0), transparent=bg)
            paste(occ, reserve, int(r0), int(c0), transparent=bg)
            bbox = (int(r0), int(c0), int(r0 + size - 1), int(c0 + size - 1))
            anchor_bboxes.append(bbox)
            objects.append(
                {
                    "kind": spec["kind"],
                    "size": size,
                    "color": color,
                    "r0": int(r0),
                    "c0": int(c0),
                    "bbox": bbox,
                    "sprite": sprite,
                }
            )
    except Exception:
        return None

    attached_count = 0
    for i, dcol in enumerate(distractor_colors):
        shape = _DISTRACTOR_SHAPES[int(np.random.randint(0, len(_DISTRACTOR_SHAPES)))]
        patch = np.full(shape, int(dcol), dtype=int)
        ok, attached = _place_distractor(g, occ, patch, bg, anchor_bboxes, force_attach=(i == 0))
        if not ok:
            return None
        if attached:
            attached_count += 1

    if attached_count < 1:
        return None

    if not _variant_visible(g):
        return None

    actual = main(g)
    bg2, collected, _distractors = _collect_components(g)
    frames, solids = _split_groups(collected)
    expected = _assemble_output(_distance_sorted(frames), _distance_sorted(solids), bg2)
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        return None

    return g


def generate_input() -> np.ndarray:
    for _ in range(900):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 900 attempts")

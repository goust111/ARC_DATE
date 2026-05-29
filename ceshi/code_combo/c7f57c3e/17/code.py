# Combo variant of c7f57c3e (index 17)
# combo_spec: D5/0+D8/2
# Name: 对齐到网格锚点 + 输出网格扩展
#
# Design analysis for the fused combo
# ----------------------------------
# spec:
#   core_preserved:
#     - 仍然只有两类多色 prototype，且每个 sprite 都是其中一类的整数倍最近邻缩放拷贝。
#     - 输出仍然把每个 sprite 换成“另一类 prototype”，并保持 scale 不变。
#     - 最小 bbox area 的 scale-1 sprites 仍然能唯一恢复这两个 prototype。
#   fused_change:
#     - D5/0: 输出位置不再沿用输入实例的原位 anchor 对齐，而是吸附到可见 guide lattice 的最近锚点。
#     - D8/2: guide lattice 不只限于输入画布内部；solver 要把 3x3 可见点阵理解成可等步长向外延拓的无限格。
#       若最近锚点落在原画布外，输出画布就朝相应方向扩展，并把扩展出来的 guide 行/列也画出来。
#   visual_behavior:
#     - 输入里固定可见的是一个 3x3 单像素 guide lattice；sprites 出现在各自最近锚点的 Voronoi cell 内，
#       因此肉眼能看出“它们靠近哪一个锚点”。
#     - 输出里 sprites 先 swap prototype，再整体平移到最近锚点；若该锚点在可见 3x3 之外，
#       输出会多出新的边缘带，里面继续出现同间距的 guide dots。
#   concrete_trace:
#     - 原题里，一个 scale-2 sprite 若被识别为 proto_A，就直接在原位置附近换成 scale-2 的 proto_B，
#       本质上由输入 sprite 的局部 anchor 决定放置位置。
#     - 融合后，同一个 sprite 先求出自己的 anchor 左上角坐标，例如它在输入里位于 row=1、col=15，
#       而 guide rows/cols 以步长 7 外推后，最近目标可能是 row=-2、col=12。于是 replacement 会被放到
#       这个外推锚点上，输出顶部新增 2 行来容纳 row=-2 的 guide row 与 sprite 本体。
#     - 差异不只是“移动到网格点”，而是“移动到可外推的网格点，并因此改变输出尺寸”。
#   complexity_calibration:
#     - guide lattice 固定为 3x3，可观察对象类别仍然少；但 solver 必须再外推一圈甚至两圈锚点，
#       足以体现 D8/2 而不会让目标空间爆炸。
#     - 仍保留 3-4 个 sprites 与一个 scale-2 个体，确保 prototype/scale 推断保留原题难度。
#     - 至少一个 scale-1 sprite 被构造成会吸附到画布外锚点，避免 generator 退化回 D5-only。
#   implicit_assumption_audit_main:
#     - 原代码 `components_mask(g != bg)` 假设所有非背景组件都是 sprites；融合后 guide dots 也是非背景，
#       所以新版必须先识别 guide color，再在 `(g != bg) & (g != guide_color)` 上找 sprites。
#     - 原代码通过“原位置 anchor 对齐 + 全局重叠打分”识别 prototype；融合后输入 sprite 本身就是完整缩放拷贝，
#       且 guide dots 不与 sprite 重叠，因此可改为按 bbox 尺寸求 scale、再做精确数组匹配。
#     - 原代码默认输出网格与输入同形；融合后不成立。新版先求每个 sprite 的目标外推锚点，再统一计算
#       `min/max row/col`，在需要的方向上扩展画布。
#     - 原代码输出从输入网格上“擦除再贴回”；融合后如果目标坐标为负数或超边界，这种做法会截断对象。
#       新版改为在新画布上重绘 guide lattice 与 replacements。
#   placement_legality:
#     - generator 先固定输入画布内的 3x3 visible lattice，再选择输出目标锚点索引，索引允许落在 `-1..3`
#       的外推范围内，但通过 bbox 检查保证输出 sprites 互不重叠。
#     - 输入位置不是随意乱放，而是从“最近外推锚点唯一等于目标索引”的整数坐标集合中挑选，
#       所以 solver 的 nearest-anchor 判定无平局。
#     - 输入与输出都显式检查 bbox 不重叠、不越界；输出验证使用 `main(inp) == expected_out`。
#   observability_and_cleanliness:
#     - guide color 与 sprite colors 分离，9 个单像素 guide dots 全部保留可见，因此人类能直接读出步长。
#     - 扩展出来的边带继续出现同间距 guide dots，而不是纯背景留白，视觉上能明确说明“画布被延伸，
#       且锚点格也被延伸”。
#     - 输入阶段不让 guide dots 被 sprites 遮住，避免把“锚点推断”和“prototype 识别”混在一起。
#   non_degeneracy_check_target:
#     - 量化指标使用 `expansion_sides = [top,left,bottom,right]` 中为真的方向数，以及
#       `offgrid_sprite_count = 目标锚点索引不在 0..2 范围内的 sprite 数量`。
#     - 若这两个量长期为 0，则样本退化回原题或 D5-only；generator 会强制至少 1 个 off-grid sprite。
#   new_main_assumptions_checked_against_generator:
#     - 假设存在唯一 guide color，且它形成完整的 3x3 等步长 singleton lattice。
#     - 假设 scale-1 sprites 中恰有两种 prototype，且所有 sprite 都与其中一个缩放版本完全一致。
#     - 假设每个输入 sprite 的 anchor 坐标到无限 lattice 的最近点唯一。
#     - 假设外推目标锚点只需落在 `-1..3` 的一圈扩展内；generator 明确按此范围构造。

from __future__ import annotations

import os
import random
import sys

import numpy as np

THIS_DIR = os.path.dirname(__file__)
SOLVER_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "..", "..", "..", "..", "solvers"))
if SOLVER_DIR not in sys.path:
    sys.path.append(SOLVER_DIR)

from arc_utils import bbox_coords, components_mask, get_bg, paste


def _pick_anchor_bbox(arr: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    ys, xs = np.where(arr == 1)
    if ys.size == 0:
        raise AssertionError("sprite array must contain color 1")
    return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())


def _scale_nn(arr: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def _nearest_lattice_index(origin: int, step: int, x: int) -> int:
    return int(np.floor(((int(x) - int(origin)) / float(step)) + 0.5))


def _detect_guide_lattice(g: np.ndarray, bg: int) -> tuple[int, list[int], list[int], int]:
    candidates: list[tuple[int, list[int], list[int], int]] = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) != bg):
        comps = components_mask(g == color, connectivity=8)
        if len(comps) != 9 or not all(len(comp) == 1 for comp in comps):
            continue
        pts = [(int(comp[0][0]), int(comp[0][1])) for comp in comps]
        rows = sorted({r for r, _c in pts})
        cols = sorted({c for _r, c in pts})
        if len(rows) != 3 or len(cols) != 3:
            continue
        row_steps = [rows[i + 1] - rows[i] for i in range(2)]
        col_steps = [cols[i + 1] - cols[i] for i in range(2)]
        if row_steps[0] <= 0 or row_steps != col_steps:
            continue
        step = int(row_steps[0])
        expected = {(r, c) for r in rows for c in cols}
        if set(pts) != expected:
            continue
        candidates.append((int(color), rows, cols, step))
    assert len(candidates) == 1, "expected exactly one complete 3x3 guide lattice"
    return candidates[0]


def _extract_sprites(g: np.ndarray, bg: int, guide_color: int) -> list[dict[str, object]]:
    mask = (g != bg) & (g != guide_color)
    comps = components_mask(mask, connectivity=8)
    assert len(comps) >= 2, "grid must contain at least 2 sprite components"

    sprites: list[dict[str, object]] = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        sprites.append({"pts": pts, "bbox": (r0, c0, r1, c1), "arr": arr})
    return sprites


def _prototype_pair(sprites: list[dict[str, object]]) -> list[np.ndarray]:
    areas = [
        int((s["bbox"][2] - s["bbox"][0] + 1) * (s["bbox"][3] - s["bbox"][1] + 1))
        for s in sprites
    ]
    min_area = min(areas)
    base = [s for s, area in zip(sprites, areas, strict=False) if area == min_area]
    uniq: dict[tuple[int, ...], np.ndarray] = {}
    for s in base:
        arr = s["arr"]
        assert isinstance(arr, np.ndarray)
        key = tuple(arr.ravel().tolist())
        if key not in uniq:
            uniq[key] = arr
    assert len(uniq) == 2, "smallest-area sprites must yield exactly 2 distinct prototypes"
    keys = list(uniq.keys())[:2]
    return [uniq[keys[0]], uniq[keys[1]]]


def _classify_sprite(arr: np.ndarray, protos: list[np.ndarray]) -> tuple[int, int]:
    base_h, base_w = protos[0].shape
    h, w = arr.shape
    assert h % base_h == 0 and w % base_w == 0, "sprite bbox must be integer-scaled prototype bbox"
    scale_h = h // base_h
    scale_w = w // base_w
    assert scale_h == scale_w and scale_h >= 1, "non-uniform scale is unsupported"
    scale = int(scale_h)

    matches = [idx for idx, proto in enumerate(protos) if np.array_equal(arr, _scale_nn(proto, scale))]
    assert len(matches) == 1, "sprite must match exactly one scaled prototype"
    return int(matches[0]), scale


def _build_output(
    input_shape: tuple[int, int],
    bg: int,
    guide_color: int,
    guide_origin_rows: list[int],
    guide_origin_cols: list[int],
    step: int,
    placements: list[dict[str, object]],
) -> np.ndarray:
    h, w = map(int, input_shape)
    row_indices = [0, 1, 2]
    col_indices = [0, 1, 2]
    for rec in placements:
        row_indices.append(int(rec["row_idx"]))
        col_indices.append(int(rec["col_idx"]))

    all_guide_rows = [int(guide_origin_rows[0] + idx * step) for idx in range(min(row_indices), max(row_indices) + 1)]
    all_guide_cols = [int(guide_origin_cols[0] + idx * step) for idx in range(min(col_indices), max(col_indices) + 1)]

    min_r, max_r = 0, h - 1
    min_c, max_c = 0, w - 1
    for r in all_guide_rows:
        min_r = min(min_r, int(r))
        max_r = max(max_r, int(r))
    for c in all_guide_cols:
        min_c = min(min_c, int(c))
        max_c = max(max_c, int(c))
    for rec in placements:
        patch = rec["patch"]
        assert isinstance(patch, np.ndarray)
        top = int(rec["top"])
        left = int(rec["left"])
        min_r = min(min_r, top)
        min_c = min(min_c, left)
        max_r = max(max_r, top + patch.shape[0] - 1)
        max_c = max(max_c, left + patch.shape[1] - 1)

    out = np.full((max_r - min_r + 1, max_c - min_c + 1), bg, dtype=int)
    r_shift = -min_r
    c_shift = -min_c

    for r in all_guide_rows:
        for c in all_guide_cols:
            out[int(r + r_shift), int(c + c_shift)] = guide_color

    for rec in placements:
        patch = rec["patch"]
        assert isinstance(patch, np.ndarray)
        paste(out, patch, int(rec["top"]) + r_shift, int(rec["left"]) + c_shift, transparent=bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap every sprite to the opposite prototype, snap it to the nearest extrapolated guide anchor, and expand the canvas only when needed.

    Input structure:
    - bg = most common color. A separate guide color forms a visible 3x3 lattice of singleton dots with uniform row/col spacing.
    - Two distinct small sprite patterns appear at integer scales, and each sprite contains color-1 anchor cells.
    - Sprites are placed inside Voronoi cells of that lattice, so some are nearest to an anchor that lies just outside the visible 3x3 block.

    Rule:
    - Recover the two scale-1 prototypes from the smallest sprites, then classify every sprite by prototype type and integer scale.
    - Treat the visible 3x3 guide lattice as part of a larger arithmetic grid that continues with the same spacing beyond the input boundary.
    - Replace each sprite with the other prototype at the same scale, align the replacement's color-1 anchor bbox top-left to the nearest extrapolated guide point, and enlarge the output grid in any direction required to contain the relocated sprites and the extended guide rows/cols.

    Key insight: The guide dots control destination geometry rather than the input sprite's original position; because the lattice is understood as extendable, the correct target anchor can sit outside the input canvas, and the output must grow to reveal that continued lattice.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    guide_color, guide_rows, guide_cols, step = _detect_guide_lattice(g, bg)
    sprites = _extract_sprites(g, bg, guide_color)
    protos = _prototype_pair(sprites)

    placements: list[dict[str, object]] = []
    for s in sprites:
        bbox = s["bbox"]
        arr = s["arr"]
        assert isinstance(bbox, tuple)
        assert isinstance(arr, np.ndarray)
        r0, c0, _r1, _c1 = bbox

        proto_idx, scale = _classify_sprite(arr, protos)
        repl_idx = 1 - proto_idx

        ar0, ac0, _ar1, _ac1 = _pick_anchor_bbox(arr, bg)
        src_anchor_r = int(r0 + ar0)
        src_anchor_c = int(c0 + ac0)
        row_idx = _nearest_lattice_index(guide_rows[0], step, src_anchor_r)
        col_idx = _nearest_lattice_index(guide_cols[0], step, src_anchor_c)
        dst_anchor_r = int(guide_rows[0] + row_idx * step)
        dst_anchor_c = int(guide_cols[0] + col_idx * step)

        repl = _scale_nn(protos[repl_idx], scale)
        pr0, pc0, _pr1, _pc1 = _pick_anchor_bbox(protos[repl_idx], bg)
        top = int(dst_anchor_r - pr0 * scale)
        left = int(dst_anchor_c - pc0 * scale)
        placements.append(
            {
                "patch": repl,
                "top": top,
                "left": left,
                "row_idx": int(row_idx),
                "col_idx": int(col_idx),
            }
        )

    return _build_output(g.shape, bg, guide_color, guide_rows, guide_cols, step, placements)


# --------------- generator (o5x) ---------------

# --- CFG: canvas ---
GRID_SIZE_MIN = 26
GRID_SIZE_MAX = 30

# --- CFG: guide lattice ---
GUIDE_SIZE = 3
GUIDE_STEP = 7
GUIDE_ROW0_CHOICES = [5, 6, 7]
GUIDE_COL0_CHOICES = [5, 6, 7]

# --- CFG: prototypes ---
PROTO_SHAPES = [(4, 3), (3, 4), (4, 4)]
BG_CHOICES = [0, 2, 3, 4, 5, 6, 7, 8, 9]
COLOR_SHARE_PROB = 0.65

# --- CFG: row-mask weights ---
ANCHOR_ROW_WEIGHTS = [("center", 0.45), ("outer", 0.35), ("full", 0.20)]
VARIABLE_ROW_WEIGHTS = [("center", 0.68), ("full", 0.32)]
OUTER_FIXUP_PROB = 0.7

# --- CFG: sprite instances ---
SPRITE_COUNTS = [3, 4]
BIG_SCALES = [2]

# --- CFG: placement ---
PLACEMENT_PADDING = 2
OUTER_SLOT_CHOICES = [
    (-1, 0), (-1, 1), (-1, 2),
    (0, -1), (1, -1), (2, -1),
    (3, 0), (3, 1), (3, 2),
    (0, 3), (1, 3), (2, 3),
]
INNER_SLOT_CHOICES = [
    (0, 0), (0, 1), (0, 2),
    (1, 0), (1, 1), (1, 2),
    (2, 0), (2, 1), (2, 2),
]

# --- CFG: generation retries ---
MAX_RETRIES = 4000

# --- Row-mask patterns indexed by width ---
_ROW_MASKS = {
    "center": {3: [0, 1, 0], 4: [0, 1, 1, 0]},
    "outer": {3: [1, 0, 1], 4: [1, 0, 0, 1]},
    "full": {3: [1, 1, 1], 4: [1, 1, 1, 1]},
}


def _row_mask(kind: str, w: int) -> list[int]:
    return list(_ROW_MASKS[kind][w])


def _anchor_bbox(arr: np.ndarray) -> tuple[int, int, int, int]:
    rs, cs = np.where(arr == 1)
    return int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())


def _scale(arr: np.ndarray, k: int) -> np.ndarray:
    return _scale_nn(arr, k)


def _make_labels() -> np.ndarray:
    h, w = random.choice(PROTO_SHAPES)

    counts = {1: 1, 2: 1, 3: 1}
    if h > 3:
        counts[random.choice([1, 2, 3])] += h - 3

    roles = []
    for role in random.sample([1, 2, 3], 3):
        roles.extend([role] * counts[role])

    masks = []
    for role in roles:
        entries = ANCHOR_ROW_WEIGHTS if role == 1 else VARIABLE_ROW_WEIGHTS
        kinds, wts = zip(*entries)
        masks.append(_row_mask(random.choices(kinds, weights=wts)[0], w))

    if not any(mask[0] or mask[-1] for mask in masks):
        idx = random.randrange(h)
        masks[idx] = _row_mask("outer" if roles[idx] == 1 and random.random() < OUTER_FIXUP_PROB else "full", w)

    if all(all(mask) for mask in masks):
        masks[random.randrange(h)] = _row_mask("center", w)

    labels = np.zeros((h, w), dtype=int)
    for r, (role, mask) in enumerate(zip(roles, masks)):
        for c, bit in enumerate(mask):
            if bit:
                labels[r, c] = role
    return labels


def _make_prototypes(bg: int) -> tuple[np.ndarray, np.ndarray]:
    labels = _make_labels()
    pool = [c for c in range(10) if c not in (bg, 1)]
    a_cols = random.sample(pool, 2)
    remaining = [c for c in pool if c not in a_cols]
    if random.random() < COLOR_SHARE_PROB:
        b_cols = [random.choice(a_cols), random.choice(remaining)]
        random.shuffle(b_cols)
    else:
        b_cols = random.sample(remaining, 2)

    def _colorize(lab: np.ndarray, cols: list[int]) -> np.ndarray:
        proto = np.full(lab.shape, bg, dtype=int)
        proto[lab == 1] = 1
        proto[lab == 2] = cols[0]
        proto[lab == 3] = cols[1]
        return proto

    return _colorize(labels, a_cols), _colorize(np.flipud(labels), b_cols)


def _bbox_with_padding(top: int, left: int, patch: np.ndarray, pad: int) -> tuple[int, int, int, int]:
    h, w = patch.shape
    return int(top - pad), int(left - pad), int(top + h - 1 + pad), int(left + w - 1 + pad)


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _cell_candidates(origin: int, step: int, limit: int, idx: int) -> list[int]:
    anchor = origin + idx * step
    vals = [x for x in range(limit) if _nearest_lattice_index(origin, step, x) == idx and x != anchor]
    random.shuffle(vals)
    vals.sort(key=lambda x: (abs(x - anchor), abs(x - (0 if idx < 0 else limit - 1))), reverse=True)
    return vals


def _slot_order(scale: int) -> list[tuple[int, int]]:
    outer = OUTER_SLOT_CHOICES[:]
    inner = INNER_SLOT_CHOICES[:]
    random.shuffle(outer)
    random.shuffle(inner)
    if scale == 1:
        return outer + inner
    return inner + outer


def _choose_output_slot(
    rows0: list[int],
    cols0: list[int],
    step: int,
    out_patch: np.ndarray,
    scale: int,
    used_boxes: list[tuple[int, int, int, int]],
    force_offgrid: bool,
) -> tuple[int, int, int, int] | None:
    pr0, pc0, _pr1, _pc1 = _anchor_bbox(out_patch)
    slots = _slot_order(scale)
    if force_offgrid:
        slots = [slot for slot in slots if slot[0] not in (0, 1, 2) or slot[1] not in (0, 1, 2)]
    for row_idx, col_idx in slots:
        ar = int(rows0[0] + row_idx * step)
        ac = int(cols0[0] + col_idx * step)
        top = int(ar - pr0)
        left = int(ac - pc0)
        box = _bbox_with_padding(top, left, out_patch, PLACEMENT_PADDING)
        if any(_bbox_overlap(box, prev) for prev in used_boxes):
            continue
        return int(row_idx), int(col_idx), top, left
    return None


def _choose_input_position(
    n: int,
    rows0: list[int],
    cols0: list[int],
    step: int,
    row_idx: int,
    col_idx: int,
    in_patch: np.ndarray,
    used_boxes: list[tuple[int, int, int, int]],
) -> tuple[int, int] | None:
    ir0, ic0, _ir1, _ic1 = _anchor_bbox(in_patch)
    row_vals = _cell_candidates(rows0[0], step, n, row_idx)
    col_vals = _cell_candidates(cols0[0], step, n, col_idx)
    for anchor_r in row_vals:
        top = int(anchor_r - ir0)
        if top < 0 or top + in_patch.shape[0] > n:
            continue
        for anchor_c in col_vals:
            left = int(anchor_c - ic0)
            if left < 0 or left + in_patch.shape[1] > n:
                continue
            if anchor_r == rows0[0] + row_idx * step and anchor_c == cols0[0] + col_idx * step:
                continue
            box = _bbox_with_padding(top, left, in_patch, PLACEMENT_PADDING)
            if any(_bbox_overlap(box, prev) for prev in used_boxes):
                continue
            return top, left
    return None


def _placement_record(
    out_type: int,
    scale: int,
    row_idx: int,
    col_idx: int,
    out_top: int,
    out_left: int,
    in_top: int,
    in_left: int,
) -> dict[str, int]:
    return {
        "out_type": int(out_type),
        "scale": int(scale),
        "row_idx": int(row_idx),
        "col_idx": int(col_idx),
        "out_top": int(out_top),
        "out_left": int(out_left),
        "in_top": int(in_top),
        "in_left": int(in_left),
    }


def _gen_attempt() -> tuple[np.ndarray, np.ndarray, dict[str, int]] | None:
    bg = random.choice(BG_CHOICES)
    proto_a, proto_b = _make_prototypes(bg)
    protos = [proto_a, proto_b]

    proto_colors = {int(c) for proto in protos for c in np.unique(proto) if int(c) != bg}
    guide_choices = [c for c in range(10) if c not in proto_colors and c != bg]
    if not guide_choices:
        return None
    guide_color = int(random.choice(guide_choices))

    count = random.choice(SPRITE_COUNTS)
    big_scale = random.choice(BIG_SCALES)
    instances = [(0, 1), (1, 1), (random.randrange(2), big_scale)]
    if count == 4:
        instances.append((random.randrange(2), 1))
    random.shuffle(instances)
    instances.sort(key=lambda item: item[1], reverse=True)

    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    row0 = random.choice(GUIDE_ROW0_CHOICES)
    col0 = random.choice(GUIDE_COL0_CHOICES)
    guide_rows = [int(row0 + i * GUIDE_STEP) for i in range(GUIDE_SIZE)]
    guide_cols = [int(col0 + i * GUIDE_STEP) for i in range(GUIDE_SIZE)]
    if guide_rows[-1] >= n or guide_cols[-1] >= n:
        return None

    out_boxes: list[tuple[int, int, int, int]] = []
    in_boxes: list[tuple[int, int, int, int]] = []
    chosen: list[dict[str, int]] = []
    offgrid_count = 0

    for idx, (out_type, scale) in enumerate(instances):
        out_patch = _scale(protos[out_type], scale)
        force_offgrid = offgrid_count == 0 and scale == 1 and all(rem_scale != 1 for _t, rem_scale in instances[idx + 1 :])
        if offgrid_count == 0 and scale == 1:
            force_offgrid = True
        slot = _choose_output_slot(guide_rows, guide_cols, GUIDE_STEP, out_patch, scale, out_boxes, force_offgrid)
        if slot is None:
            return None
        row_idx, col_idx, out_top, out_left = slot
        out_box = _bbox_with_padding(out_top, out_left, out_patch, PLACEMENT_PADDING)
        out_boxes.append(out_box)

        in_patch = _scale(protos[1 - out_type], scale)
        inp_pos = _choose_input_position(n, guide_rows, guide_cols, GUIDE_STEP, row_idx, col_idx, in_patch, in_boxes)
        if inp_pos is None:
            return None
        in_top, in_left = inp_pos
        in_box = _bbox_with_padding(in_top, in_left, in_patch, PLACEMENT_PADDING)
        in_boxes.append(in_box)

        if row_idx not in (0, 1, 2) or col_idx not in (0, 1, 2):
            offgrid_count += 1

        chosen.append(_placement_record(out_type, scale, row_idx, col_idx, out_top, out_left, in_top, in_left))

    if offgrid_count == 0:
        return None

    inp = np.full((n, n), bg, dtype=int)
    for r in guide_rows:
        for c in guide_cols:
            inp[r, c] = guide_color
    for rec in chosen:
        patch = _scale(protos[1 - rec["out_type"]], rec["scale"])
        paste(inp, patch, rec["in_top"], rec["in_left"], transparent=bg)

    placements: list[dict[str, object]] = []
    row_indices = [0, 1, 2]
    col_indices = [0, 1, 2]
    for rec in chosen:
        patch = _scale(protos[rec["out_type"]], rec["scale"])
        placements.append(
            {
                "patch": patch,
                "top": rec["out_top"],
                "left": rec["out_left"],
                "row_idx": rec["row_idx"],
                "col_idx": rec["col_idx"],
            }
        )
        row_indices.append(rec["row_idx"])
        col_indices.append(rec["col_idx"])
    expected = _build_output(inp.shape, bg, guide_color, guide_rows, guide_cols, GUIDE_STEP, placements)

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None

    stats = {
        "offgrid_sprite_count": int(offgrid_count),
        "expanded_top": int(min(row_indices) < 0),
        "expanded_left": int(min(col_indices) < 0),
        "expanded_bottom": int(max(row_indices) > 2),
        "expanded_right": int(max(col_indices) > 2),
    }
    return inp, expected, stats


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        pack = _gen_attempt()
        if pack is not None:
            inp, _out, _stats = pack
            return inp
    raise RuntimeError("failed to generate a valid c7f57c3e combo input")

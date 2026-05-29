# Variant of c7f57c3e
# Dimension: D5 — 目标位置/锚定方式 (dst)
# Change: 交换后精灵保持在与配对精灵最大重叠的位置，本质上近似原位交换... → 对齐到网格锚点
#
# Design analysis for the D5 variant
# ----------------------------------
# spec:
#   core_preserved:
#     - 仍然只有两类多色 prototype。
#     - 每个 sprite 仍然是其中一类的整数倍最近邻缩放拷贝。
#     - 输出仍然把每个 sprite 换成“另一类 prototype”，并保持 scale 不变。
#   D5_change:
#     - 原题输出位置由“与当前实例内部锚点对齐，因此近似原位替换”决定。
#     - 本变体改为“与最近的预定义网格锚点对齐”。
#     - 这些锚点在输入中可见：它们是一个独立颜色构成的 3x3 稀疏点阵。
#     - 每个 sprite 的内部 color-1 anchor bbox 的左上角，是被吸附到最近点阵交点的对象锚点。
#   visual_behavior:
#     - 输入里：锚点点阵固定不动，sprites 出现在点阵附近但不严格对齐。
#     - 输出里：sprites 先 swap 成另一 prototype，再整体平移到最近锚点。
#     - 少量锚点可能被 sprite 覆盖，但仍保留足够多的点让人看出规则。
#   concrete_trace:
#     - 原逻辑：某个 scale-2 的 proto_A 实例在 (r,c) 一带，被识别后直接换成 scale-2 的 proto_B，
#       放置时继续让 color-1 子区域对齐到原实例的同一局部位置，因此几乎不动。
#     - 新逻辑：同一个实例先识别为 proto_A / scale-2，再找出它的 color-1 anchor 左上角坐标，
#       把这个坐标吸附到最近的网格点，例如从 (14,17) 吸附到 (13,21)，然后把 scale-2 的 proto_B
#       整体平移过去。差异是输出位置受“最近锚点”控制，而不再由输入原位控制。
#   complexity_calibration:
#     - 点阵固定为 3x3，避免目标类型太多导致归纳困难，同时有 9 个可选锚点，足够支撑 3-4 个 sprites。
#     - 仍保留两类 prototype 与 scale 变化，但把大尺度限制到 2，避免稀疏锚点被大对象完全淹没。
#     - 锚点颜色不参与任何排序或变换，只作为显式几何参照。
#   implicit_assumption_audit_main:
#     - 原代码 `components_mask(g != bg)` 默认所有非背景组件都是 sprites；这里不成立，因为锚点也是非背景。
#       新版先识别 guide color，再在 `(g != bg) & (g != guide_color)` 上找 sprites。
#     - 原代码从全网格重叠打分来判断 prototype；这里 guide dots 可能落在 sprite bbox 的洞里，
#       所以新版在提取局部 patch 时先把 guide color 还原成 bg，再做局部精确匹配。
#     - 原代码的放置位置来自原实例 anchor；这里改为最近 guide point，因此所有 `top/left` 计算都改成
#       “目标锚点 - replacement prototype 的 anchor 偏移”。
#   placement_legality:
#     - generator 先固定 3x3 guide lattice，再为输出选择互不冲突的 anchor。
#     - 输入实例只在各自 anchor 的 Voronoi cell 内偏移，因此 solver 的最近锚点判定唯一。
#     - 生成后检查每一行/列锚点至少还可见一个，保证 solver 能恢复整张 lattice。
#   observability_and_cleanliness:
#     - 人类可直接看到一个静止点阵，以及 sprites 从“近旁位置”移动到“点阵对齐位置”。
#     - guide color 独立于 sprite 颜色，视觉上可分层，不需要依赖颜色 id 的隐藏语义。
#   new_main_assumptions_checked_against_generator:
#     - 假设存在唯一的 guide color，且其可见单像素点落在 3 行 3 列等步长坐标上。
#     - 假设最小 bbox area 的 sprite 中正好能恢复两个 scale-1 prototypes。
#     - 假设每个输入 sprite 到其目标锚点的最近关系无平局。
#     - 这些条件都由 generator 明确构造并在生成后验证。

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
    """Pick the color-1 anchor bbox inside a sprite array."""
    ys, xs = np.where(arr == 1)
    if ys.size == 0:
        raise AssertionError("sprite array must contain color 1")
    return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())


def _scale_nn(arr: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def _nearest_value(values: list[int], x: int) -> int:
    return min(values, key=lambda v: (abs(int(v) - int(x)), int(v)))


def _detect_guide_lattice(g: np.ndarray, bg: int) -> tuple[int, list[int], list[int]]:
    """Find the unique guide-dot color and the 3x3 lattice row/col coordinates."""
    candidates: list[tuple[int, int, list[int], list[int]]] = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) != bg):
        comps = components_mask(g == color, connectivity=8)
        if not comps or not all(len(comp) == 1 for comp in comps):
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
        if len(pts) < 5:
            continue
        candidates.append((len(pts), color, rows, cols))
    assert len(candidates) == 1, "expected exactly one 3x3 guide lattice color"
    _count, color, rows, cols = candidates[0]
    return int(color), rows, cols


def _extract_sprites(g: np.ndarray, bg: int, guide_color: int) -> list[dict[str, object]]:
    mask = (g != bg) & (g != guide_color)
    comps = components_mask(mask, connectivity=8)
    assert len(comps) >= 2, "grid must contain at least 2 sprite components"

    sprites: list[dict[str, object]] = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        arr[arr == guide_color] = bg
        sprites.append({"pts": pts, "bbox": (r0, c0, r1, c1), "arr": arr})
    return sprites


def _prototype_pair(sprites: list[dict[str, object]]) -> list[np.ndarray]:
    areas = [
        int((s["bbox"][2] - s["bbox"][0] + 1) * (s["bbox"][3] - s["bbox"][1] + 1))
        for s in sprites
    ]
    min_area = min(areas)
    base = [s for s, a in zip(sprites, areas, strict=False) if a == min_area]
    uniq: dict[tuple[int, ...], np.ndarray] = {}
    for s in base:
        arr = s["arr"]
        assert isinstance(arr, np.ndarray)
        key = tuple(arr.ravel().tolist())
        if key not in uniq:
            uniq[key] = arr
    assert len(uniq) == 2, "smallest-area sprites must yield exactly 2 distinct prototypes"
    proto_keys = list(uniq.keys())[:2]
    return [uniq[proto_keys[0]], uniq[proto_keys[1]]]


def _classify_sprite(arr: np.ndarray, protos: list[np.ndarray], bg: int) -> tuple[int, int]:
    base_h, base_w = protos[0].shape
    h, w = arr.shape
    assert h % base_h == 0 and w % base_w == 0, "sprite must be an integer scale of prototype bbox"
    scale_h = h // base_h
    scale_w = w // base_w
    assert scale_h == scale_w and scale_h >= 1, "non-uniform scale is unsupported"
    scale = int(scale_h)

    matches = [idx for idx, p in enumerate(protos) if np.array_equal(arr, _scale_nn(p, scale))]
    assert len(matches) == 1, "sprite must match exactly one scaled prototype"
    proto_idx = int(matches[0])

    anc = _pick_anchor_bbox(arr, bg)
    ar0, ac0, _ar1, _ac1 = anc
    return proto_idx, scale


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap every sprite to the opposite prototype and snap it onto the nearest guide anchor.

    Input structure:
    - bg = most common color. A separate guide color forms a sparse 3×3 lattice of single-pixel anchors.
    - Two distinct small sprite patterns appear at integer scales, and each sprite contains color-1 anchor cells.
    - Sprites sit near the guide lattice rather than exactly on it.

    Rule:
    - Recover the two scale-1 prototypes from the smallest sprites.
    - For each sprite, identify its prototype class and scale from its local pattern.
    - Replace it with the other prototype at the same scale, then move it so the sprite's color-1 anchor bbox top-left snaps to the nearest guide-lattice point.

    Key insight: The swap rule is unchanged, but position is no longer inherited from the input sprite; the visible guide lattice supplies the destination geometry, so each replacement is relocated by nearest-anchor snapping.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    guide_color, guide_rows, guide_cols = _detect_guide_lattice(g, bg)
    sprites = _extract_sprites(g, bg, guide_color)
    protos = _prototype_pair(sprites)

    out = g.copy()

    for s in sprites:
        bbox = s["bbox"]
        arr = s["arr"]
        assert isinstance(bbox, tuple)
        assert isinstance(arr, np.ndarray)
        r0, c0, _r1, _c1 = bbox

        proto_idx, scale = _classify_sprite(arr, protos, bg)
        repl_idx = 1 - proto_idx

        anc = _pick_anchor_bbox(arr, bg)
        ar0, ac0, _ar1, _ac1 = anc
        src_anchor_r = int(r0 + ar0)
        src_anchor_c = int(c0 + ac0)
        dst_anchor_r = _nearest_value(guide_rows, src_anchor_r)
        dst_anchor_c = _nearest_value(guide_cols, src_anchor_c)

        for rr, cc in s["pts"]:
            out[int(rr), int(cc)] = bg

        repl = _scale_nn(protos[repl_idx], scale)
        pr0, pc0, _pr1, _pc1 = _pick_anchor_bbox(protos[repl_idx], bg)
        top = int(dst_anchor_r - pr0 * scale)
        left = int(dst_anchor_c - pc0 * scale)
        assert 0 <= top and 0 <= left and top + repl.shape[0] <= h and left + repl.shape[1] <= w
        paste(out, repl, top, left, transparent=bg)

    return out


# --------------- generator (o5x) ---------------

# --- CFG: canvas ---
GRID_SIZE_MIN = 25
GRID_SIZE_MAX = 29
GRID_MARGIN_FOR_SPRITES = 6

# --- CFG: prototypes ---
PROTO_SHAPES = [(4, 3), (3, 4), (4, 4)]
BG_CHOICES = [0, 2, 3, 4, 5, 6, 7, 8, 9]
COLOR_SHARE_PROB = 0.65

# --- CFG: row-mask weights ---
ANCHOR_ROW_WEIGHTS = [("center", 1.0)]
VARIABLE_ROW_WEIGHTS = [("center", 0.55), ("full", 0.30), ("outer", 0.15)]
OUTER_FIXUP_PROB = 0.7

# --- CFG: sprite instances ---
SPRITE_COUNTS = [3, 4]
BIG_SCALES = [2]

# --- CFG: guide lattice ---
GUIDE_SIZE = 3
GUIDE_STEP = 8
GUIDE_VISIBLE_MIN = 5

# --- CFG: placement ---
PLACEMENT_PADDING = 2
INPUT_ANCHOR_OFFSETS = [
    (-3, -2), (-3, 1), (-2, 3), (1, -3), (2, -3), (3, 2), (3, -1), (-1, 3),
]

# --- CFG: generation retries ---
MAX_RETRIES = 400

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
    """Build an (h, w) label grid with roles 1 (anchor), 2, 3 and bg holes (0)."""
    for _ in range(200):
        h, w = random.choice(PROTO_SHAPES)

        counts = {1: 1, 2: 1, 3: 1}
        if h > 3:
            counts[random.choice([2, 3])] += h - 3

        roles = []
        for role in random.sample([1, 2, 3], 3):
            roles.extend([role] * counts[role])

        masks = []
        for role in roles:
            entries = ANCHOR_ROW_WEIGHTS if role == 1 else VARIABLE_ROW_WEIGHTS
            kinds, wts = zip(*entries)
            chosen_kind = random.choices(kinds, weights=wts)[0]
            masks.append(_row_mask(chosen_kind, w))

        if not any(mask[0] or mask[-1] for mask in masks):
            i = random.randrange(h)
            if roles[i] == 1 and random.random() < OUTER_FIXUP_PROB:
                masks[i] = _row_mask("outer", w)
            else:
                masks[i] = _row_mask("full", w)

        if all(all(mask) for mask in masks):
            i = random.randrange(h)
            if roles[i] == 1:
                masks[i] = _row_mask("center", w)
            else:
                masks[i] = _row_mask("outer", w)

        labels = np.zeros((h, w), dtype=int)
        for r, (role, mask) in enumerate(zip(roles, masks)):
            for c, bit in enumerate(mask):
                if bit:
                    labels[r, c] = role

        ar0, ac0, ar1, ac1 = _anchor_bbox(np.where(labels == 1, 1, 0))
        if ar0 >= 1 and ar1 <= h - 2 and ac0 >= 1 and ac1 <= w - 2:
            return labels
    raise RuntimeError("failed to make labels with an inset anchor")


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

    proto_a = _colorize(labels, a_cols)
    proto_b = _colorize(np.flipud(labels), b_cols)
    return proto_a, proto_b


def _bbox_with_padding(r0: int, c0: int, patch: np.ndarray, pad: int) -> tuple[int, int, int, int]:
    h, w = patch.shape
    return int(r0 - pad), int(c0 - pad), int(r0 + h - 1 + pad), int(c0 + w - 1 + pad)


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _anchor_positions(n: int) -> tuple[list[int], list[int]]:
    r0 = random.randint(4, n - 1 - 2 * GUIDE_STEP - 4)
    c0 = random.randint(4, n - 1 - 2 * GUIDE_STEP - 4)
    rows = [int(r0 + i * GUIDE_STEP) for i in range(GUIDE_SIZE)]
    cols = [int(c0 + i * GUIDE_STEP) for i in range(GUIDE_SIZE)]
    return rows, cols


def _covered_guides(
    patch: np.ndarray, top: int, left: int, guide_points: list[tuple[int, int]]
) -> set[tuple[int, int]]:
    covered: set[tuple[int, int]] = set()
    h, w = patch.shape
    for gr, gc in guide_points:
        if top <= gr < top + h and left <= gc < left + w:
            covered.add((int(gr), int(gc)))
    return covered


def _visible_guide_ok(
    covered: set[tuple[int, int]], guide_rows: list[int], guide_cols: list[int]
) -> bool:
    visible = {(r, c) for r in guide_rows for c in guide_cols if (r, c) not in covered}
    if len(visible) < GUIDE_VISIBLE_MIN:
        return False
    return (
        len({r for r, _c in visible}) == len(guide_rows)
        and len({c for _r, c in visible}) == len(guide_cols)
    )


def _candidate_anchor_slots(
    guide_rows: list[int],
    guide_cols: list[int],
    patch: np.ndarray,
    n: int,
    used_boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    pr0, pc0, _pr1, _pc1 = _anchor_bbox(patch)
    h, w = patch.shape
    slots: list[tuple[int, int, int, int]] = []
    for ar in guide_rows:
        for ac in guide_cols:
            top = int(ar - pr0)
            left = int(ac - pc0)
            if top < 0 or left < 0 or top + h > n or left + w > n:
                continue
            box = _bbox_with_padding(top, left, patch, PLACEMENT_PADDING)
            if any(_bbox_overlap(box, prev) for prev in used_boxes):
                continue
            slots.append((int(ar), int(ac), top, left))
    random.shuffle(slots)
    return slots


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    bg = random.choice(BG_CHOICES)
    proto_a, proto_b = _make_prototypes(bg)
    protos = [proto_a, proto_b]

    proto_colors = {int(c) for p in protos for c in np.unique(p) if int(c) != bg}
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

    max_dim = max(max(protos[typ].shape) * scale for typ, scale in instances)
    n = random.randint(max(GRID_SIZE_MIN, max_dim + GRID_MARGIN_FOR_SPRITES), GRID_SIZE_MAX)
    guide_rows, guide_cols = _anchor_positions(n)
    guide_points = [(r, c) for r in guide_rows for c in guide_cols]

    out = np.full((n, n), bg, dtype=int)
    inp = np.full((n, n), bg, dtype=int)
    for r, c in guide_points:
        out[r, c] = guide_color
        inp[r, c] = guide_color

    output_boxes: list[tuple[int, int, int, int]] = []
    chosen: list[dict[str, object]] = []

    for out_type, scale in instances:
        output_patch = _scale(protos[out_type], scale)
        slots = _candidate_anchor_slots(guide_rows, guide_cols, output_patch, n, output_boxes)
        if not slots:
            return None
        ar, ac, out_r, out_c = slots[0]
        output_boxes.append(_bbox_with_padding(out_r, out_c, output_patch, PLACEMENT_PADDING))
        chosen.append(
            {
                "out_type": int(out_type),
                "scale": int(scale),
                "anchor": (int(ar), int(ac)),
                "out_top": int(out_r),
                "out_left": int(out_c),
            }
        )

    input_boxes: list[tuple[int, int, int, int]] = []
    covered_guides: set[tuple[int, int]] = set()

    for rec in chosen:
        out_type = int(rec["out_type"])
        scale = int(rec["scale"])
        ar, ac = rec["anchor"]
        input_patch = _scale(protos[1 - out_type], scale)
        ir0, ic0, _ir1, _ic1 = _anchor_bbox(input_patch)

        offsets = INPUT_ANCHOR_OFFSETS[:]
        random.shuffle(offsets)
        placed = False
        for dr, dc in offsets:
            in_anchor_r = int(ar + dr)
            in_anchor_c = int(ac + dc)
            if _nearest_value(guide_rows, in_anchor_r) != ar or _nearest_value(guide_cols, in_anchor_c) != ac:
                continue
            top = int(in_anchor_r - ir0)
            left = int(in_anchor_c - ic0)
            if top < 0 or left < 0 or top + input_patch.shape[0] > n or left + input_patch.shape[1] > n:
                continue
            box = _bbox_with_padding(top, left, input_patch, PLACEMENT_PADDING)
            if any(_bbox_overlap(box, prev) for prev in input_boxes):
                continue
            tentative_covered = covered_guides | _covered_guides(input_patch, top, left, guide_points)
            if not _visible_guide_ok(tentative_covered, guide_rows, guide_cols):
                continue

            rec["in_top"] = int(top)
            rec["in_left"] = int(left)
            input_boxes.append(box)
            covered_guides = tentative_covered
            placed = True
            break
        if not placed:
            return None

    for rec in chosen:
        out_type = int(rec["out_type"])
        scale = int(rec["scale"])
        out_patch = _scale(protos[out_type], scale)
        in_patch = _scale(protos[1 - out_type], scale)
        paste(out, out_patch, int(rec["out_top"]), int(rec["out_left"]), transparent=bg)
        paste(inp, in_patch, int(rec["in_top"]), int(rec["in_left"]), transparent=bg)

    try:
        solved = main(inp)
    except Exception:
        return None
    if np.array_equal(solved, out):
        return inp, out
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        pair = _gen_attempt()
        if pair is not None:
            grid, _out = pair
            return grid
    raise RuntimeError("failed to generate a valid c7f57c3e D5 input")

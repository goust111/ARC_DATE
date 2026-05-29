# Variant of c7f57c3e
# Dimension: D2 — 配对匹配策略 (match_cond)
# Change: 跨组配对时要求尺度相同（s.scale == d.scale）且位置重叠最大化... → 按边缘颜色匹配

import random
from collections import defaultdict

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis
# ---------------
# variant_spec:
#   input_concretization:
#     - 每个 sprite 仍然是 8 连通、多色、可按整数倍缩放的精灵。
#     - 与原题不同的是：每个 prototype 的最上边缘固定带一整行统一颜色的 tag，
#       这行颜色就是 edge_color；它既属于 sprite 外边缘，又不参与 anchor。
#     - 每个网格固定出现 2 个 edge_color；每个 edge_color 下有 2 个最小原型，
#       分别代表“组 A / 组 B”。因此最小面积原型从原题的 2 个变为 4 个。
#     - 大 sprite 仍然是这些最小原型的整数倍放大拷贝，所以视觉上仍保留原题的
#       “小样本 + 一个放大体”结构，但配对线索来自外边缘标签而不是尺度重叠。
#   behavior_trace:
#     - 原逻辑：识别 2 个 prototype -> 用同尺度下与哪一个重叠分数更高来判断当前属于哪组
#       -> 换成另一组 prototype。
#     - 新逻辑：识别 4 个最小 prototype -> 先读出 edge_color，把 prototype 按 edge_color
#       分成两个二元桶 -> 每个 sprite 先匹配到自己的缩放原型，再在“同 edge_color 的桶里”
#       取另一个 prototype 作为替换对象。
#     - 具体例子：若最小原型中出现 `红边+A`、`红边+B`、`黄边+A`、`黄边+B`，
#       那么一个放大的 `红边+A` 只能变成 `红边+B`，不能变成 `黄边+B`。
#   complexity_calibration:
#     - edge_color 只取 2 档，避免自由度过多；但每档都在 A/B 两组中重复出现，
#       让解题者能从少量样例观察到“同边缘色成对”的规律。
#     - 输出里仍保留原题的尺度保持、anchor 对齐与 prototype 互换，因此变化集中在 D2。
#     - 若忽略 edge_color，只按“换到另一组”理解，会在两种边缘标签之间产生歧义；
#       generator 会让同组不同 edge_color 的内部配色不同，保证这种歧义是可见的。
#   implicit_assumptions_review:
#     - 原 `len(uniq) == 2` 依赖“最小原型只有两种”，在本变体中失效，必须改为 4。
#     - 原 `repl_idx = 1 - best_idx` 依赖全局只有一对 prototype，本变体必须改为
#       “edge_color 桶内的另一个原型”。
#     - 原 overlap score 只是用来区分 A/B；本变体若继续用它，会把不同 edge_color 的
#       同组原型混在一起，因此改成“先按整数尺度精确匹配缩放原型，再按 edge_color 配对”。
#     - 原生成器默认 1 个大 sprite + 2 个最小 prototype 就够了；现在若不把 4 个最小
#       原型都放出来，solver 虽可运行，但人类无法从样例推出标签配对关系，因此 generator
#       必须显式放置全部 4 个 scale-1 原型。
#   placement_legality:
#     - 仍然使用 output-first / oracle-check：先选定输入 sprite 列表，再由 solver 规则
#       推导 output；放置时对每个 input/output patch 计算 anchor 对齐后的并集 bbox，
#       用该 bbox 预留空间，保证输入输出都不越界且互不重叠。
#     - 所有 sprite 通过 `random_free_location` 放置，保留边框和 padding，避免组件粘连。
#   visual_observability:
#     - 外边缘 tag 是“一整行统一颜色”，人眼很容易把它识别成标签。
#     - 两个 edge_color 在每个网格中都重复出现至少两次（A/B 各一次），大 sprite 的输出
#       必须引用同标签的小样本，因此规则可从 2-3 个样例中归纳。
#   visual_tidiness:
#     - tag 只占最上方一行，不与内部锚点或孔洞叠加；对象之间维持固定间隔，避免视觉混杂。
#   color_inference:
#     - 颜色变化并非按颜色 id 排序；edge_color 只是边界标签，内部颜色由同标签的对组原型
#       提供参照。每个网格只使用 2 个 tag 色和 4 个内部色，颜色种类受控且有结构线索。
#   new_main_assumptions:
#     - 最小面积层必须恰好恢复 4 个不同原型。
#     - 每个原型最上边一整行必须是统一的非 bg、非 1 标签色。
#     - 每个 edge_color 恰好对应 2 个原型。
#     - 所有大 sprite 都是这些最小原型的 NN 整数倍缩放；generator 会保证这一点。


def _pick_anchor_bbox(arr: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    """Pick a stable anchor bbox inside the sprite array (color-1 pixels; always present)."""
    if np.any(arr == 1):
        ys, xs = np.where(arr == 1)
        return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())
    raise AssertionError("sprite array has no color-1 pixels and should always have them")


def _scale_nn(arr: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def _edge_color(arr: np.ndarray, bg: int) -> int:
    """Read the special outer-edge tag row (always the top row in generated data)."""
    top = arr[0]
    col = int(top[0])
    assert col not in (int(bg), 1), "top edge row must be a non-bg, non-anchor tag color"
    assert np.all(top == col), "top edge row must be uniform"
    return col


def _infer_scale(arr: np.ndarray, base_anchor_h: int, base_anchor_w: int, bg: int) -> int:
    ar0, ac0, ar1, ac1 = _pick_anchor_bbox(arr, bg)
    inst_anc_h = int(ar1 - ar0 + 1)
    inst_anc_w = int(ac1 - ac0 + 1)
    assert inst_anc_h % int(base_anchor_h) == 0 and inst_anc_w % int(base_anchor_w) == 0
    scale_h = inst_anc_h // int(base_anchor_h)
    scale_w = inst_anc_w // int(base_anchor_w)
    assert scale_h == scale_w and scale_h >= 1
    return int(scale_h)


def _match_scaled_proto(arr: np.ndarray, protos: list[np.ndarray], scale: int) -> int:
    for idx, proto in enumerate(protos):
        cand = _scale_nn(proto, scale)
        if cand.shape == arr.shape and np.array_equal(cand, arr):
            return int(idx)
    raise AssertionError("sprite does not match any scaled prototype")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap every sprite to the opposite prototype that shares its edge-color tag.

    Input structure:
    - bg = most common color. Small sprites reveal 4 base prototypes: 2 edge-color families, each family containing 2 opposite-group patterns.
    - Every sprite has a full top edge row in a special color; larger sprites are integer NN-scaled copies of one of those tagged prototypes.

    Rule:
    - Recover the 4 smallest prototypes and group them by their top-edge tag color.
    - For each sprite, infer its integer scale and identify which tagged prototype it is a copy of.
    - Replace it with the other prototype from the same edge-color family, keeping the same scale and anchor alignment.

    Key insight: The outer edge row is the pairing key. Scale still matters for resizing, but the correct cross-group swap is determined by matching edge-color families before toggling to the companion prototype.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    comps = components_mask(g != bg, connectivity=8)
    assert len(comps) >= 5, "grid must contain at least 5 sprite components"

    sprites = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        sprites.append({"pts": pts, "bbox": (r0, c0, r1, c1), "arr": arr})

    areas = [int((s["bbox"][2] - s["bbox"][0] + 1) * (s["bbox"][3] - s["bbox"][1] + 1)) for s in sprites]
    min_area = min(areas)
    base = [s for s, a in zip(sprites, areas) if a == min_area]
    uniq: dict[tuple[int, ...], np.ndarray] = {}
    for s in base:
        key = tuple(s["arr"].ravel().tolist())
        if key not in uniq:
            uniq[key] = s["arr"]
    assert len(uniq) == 4, "smallest-area sprites must yield exactly 4 distinct tagged prototypes"

    protos = list(uniq.values())
    edge_buckets: dict[int, list[int]] = defaultdict(list)
    for idx, proto in enumerate(protos):
        edge_buckets[_edge_color(proto, bg)].append(int(idx))
    assert len(edge_buckets) == 2, "there must be exactly 2 edge-color families"
    assert all(len(idxs) == 2 for idxs in edge_buckets.values()), "each edge-color family must contain 2 prototypes"

    anc0 = _pick_anchor_bbox(protos[0], bg)
    base_anchor_h = int(anc0[2] - anc0[0] + 1)
    base_anchor_w = int(anc0[3] - anc0[1] + 1)

    out = g.copy()

    for s in sprites:
        arr = s["arr"]
        scale = _infer_scale(arr, base_anchor_h, base_anchor_w, bg)
        cur_idx = _match_scaled_proto(arr, protos, scale)
        edge = _edge_color(protos[cur_idx], bg)
        mates = edge_buckets[edge]
        repl_idx = int(mates[0] if mates[1] == cur_idx else mates[1])

        for rr, cc in s["pts"]:
            out[int(rr), int(cc)] = bg

        r0, c0, _r1, _c1 = s["bbox"]
        ar0, ac0, _ar1, _ac1 = _pick_anchor_bbox(arr, bg)
        p = protos[repl_idx]
        sp = _scale_nn(p, scale)
        pr0, pc0, _pr1, _pc1 = _pick_anchor_bbox(p, bg)
        top = int(r0 + ar0 - pr0 * scale)
        left = int(c0 + ac0 - pc0 * scale)
        paste(out, sp, top, left, transparent=bg)

    return out


# --------------- generator (o5x) ---------------

# --- CFG: canvas ---
GRID_SIZE_MIN = 20
GRID_SIZE_MAX = 30
GRID_MARGIN_FOR_SPRITES = 8

# --- CFG: prototypes ---
CORE_SHAPES = [(3, 3), (3, 4), (4, 4)]
BG_CHOICES = [0, 2, 3, 4, 5, 6, 7, 8, 9]
PAIR_SCHEMES = [
    ((0, 1), (2, 3), (0, 2), (1, 3)),
    ((0, 1), (2, 3), (0, 3), (1, 2)),
    ((0, 2), (1, 3), (0, 1), (2, 3)),
]

# --- CFG: row-mask weights ---
ANCHOR_ROW_WEIGHTS = [("center", 0.45), ("outer", 0.35), ("full", 0.20)]
VARIABLE_ROW_WEIGHTS = [("center", 0.68), ("full", 0.32)]
OUTER_FIXUP_PROB = 0.7

# --- CFG: sprite instances ---
BIG_SCALES = [2, 3]

# --- CFG: placement ---
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2
PLACEMENT_CONNECTIVITY = 8

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
    if k == 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, k, axis=0), k, axis=1)


def _make_labels() -> np.ndarray:
    """Build a core label grid with roles 1 (anchor), 2, 3 and bg holes (0)."""
    h, w = random.choice(CORE_SHAPES)

    counts = {1: 1, 2: 1, 3: 1}
    if h > 3:
        counts[random.choice([1, 2, 3])] += h - 3
    roles = []
    for role in random.sample([1, 2, 3], 3):
        roles.extend([role] * counts[role])

    weights_by_role = {
        1: ANCHOR_ROW_WEIGHTS,
        2: VARIABLE_ROW_WEIGHTS,
        3: VARIABLE_ROW_WEIGHTS,
    }
    masks = []
    for role in roles:
        kinds, wts = zip(*weights_by_role[role])
        chosen_kind = random.choices(kinds, weights=wts)[0]
        masks.append(_row_mask(chosen_kind, w))

    if not any(mask[0] or mask[-1] for mask in masks):
        i = random.randrange(h)
        if roles[i] == 1 and random.random() < OUTER_FIXUP_PROB:
            masks[i] = _row_mask("outer", w)
        else:
            masks[i] = _row_mask("full", w)

    if all(all(mask) for mask in masks):
        masks[random.randrange(h)] = _row_mask("center", w)

    labels = np.zeros((h, w), dtype=int)
    for r, (role, mask) in enumerate(zip(roles, masks)):
        for c, bit in enumerate(mask):
            if bit:
                labels[r, c] = role

    assert np.any(labels == 1)
    return labels


def _colorize_core(labels: np.ndarray, cols: tuple[int, int], bg: int) -> np.ndarray:
    core = np.full(labels.shape, bg, dtype=int)
    core[labels == 1] = 1
    core[labels == 2] = int(cols[0])
    core[labels == 3] = int(cols[1])
    return core


def _tagged_proto(core: np.ndarray, edge_color: int) -> np.ndarray:
    edge = np.full((1, core.shape[1]), int(edge_color), dtype=int)
    return np.vstack([edge, core])


def _make_prototypes(bg: int) -> tuple[list[np.ndarray], dict[int, int]]:
    """Create 4 tagged prototypes: 2 edge-color families x 2 opposite-group patterns."""
    labels = _make_labels()
    flipped = np.flipud(labels)

    pool = [c for c in range(10) if c not in (bg, 1)]
    edge_colors = random.sample(pool, 2)
    remaining = [c for c in pool if c not in edge_colors]
    var_colors = random.sample(remaining, 4)
    scheme = random.choice(PAIR_SCHEMES)
    pairs = [tuple(var_colors[i] for i in pair) for pair in scheme]

    proto_a0 = _tagged_proto(_colorize_core(labels, pairs[0], bg), edge_colors[0])
    proto_b0 = _tagged_proto(_colorize_core(flipped, pairs[1], bg), edge_colors[0])
    proto_a1 = _tagged_proto(_colorize_core(labels, pairs[2], bg), edge_colors[1])
    proto_b1 = _tagged_proto(_colorize_core(flipped, pairs[3], bg), edge_colors[1])

    protos = [proto_a0, proto_b0, proto_a1, proto_b1]
    pair_map = {0: 1, 1: 0, 2: 3, 3: 2}
    return protos, pair_map


def _aligned_union_offsets(
    output_patch: np.ndarray, input_patch: np.ndarray
) -> tuple[int, int, int, int, int, int]:
    """
    Return output/input offsets inside a reserved union box when anchors are aligned.

    The returned tuple is:
      (out_r, out_c, in_r, in_c, union_h, union_w)
    """
    oa = _anchor_bbox(output_patch)
    ia = _anchor_bbox(input_patch)
    raw_in_r = int(oa[0] - ia[0])
    raw_in_c = int(oa[1] - ia[1])

    top = min(0, raw_in_r)
    left = min(0, raw_in_c)
    bottom = max(int(output_patch.shape[0]), raw_in_r + int(input_patch.shape[0]))
    right = max(int(output_patch.shape[1]), raw_in_c + int(input_patch.shape[1]))

    out_r = int(-top)
    out_c = int(-left)
    in_r = int(raw_in_r - top)
    in_c = int(raw_in_c - left)
    return out_r, out_c, in_r, in_c, int(bottom - top), int(right - left)


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_CHOICES)
    protos, pair_map = _make_prototypes(bg)

    big_scale = random.choice(BIG_SCALES)
    instances = [(idx, 1) for idx in range(4)] + [(random.randrange(4), big_scale)]

    patch_specs = []
    max_union_dim = 0
    for input_idx, scale in instances:
        input_patch = _scale(protos[input_idx], scale)
        output_patch = _scale(protos[pair_map[input_idx]], scale)
        off = _aligned_union_offsets(output_patch, input_patch)
        out_r, out_c, in_r, in_c, uh, uw = off
        max_union_dim = max(max_union_dim, uh, uw)
        patch_specs.append(
            {
                "scale": scale,
                "input_patch": input_patch,
                "output_patch": output_patch,
                "out_r": out_r,
                "out_c": out_c,
                "in_r": in_r,
                "in_c": in_c,
                "uh": uh,
                "uw": uw,
            }
        )

    low = max(GRID_SIZE_MIN, max_union_dim + GRID_MARGIN_FOR_SPRITES)
    if low > GRID_SIZE_MAX:
        return None
    n = random.randint(low, GRID_SIZE_MAX)
    inp = np.full((n, n), bg, dtype=int)
    out = np.full((n, n), bg, dtype=int)
    reserve = np.full((n, n), bg, dtype=int)

    patch_specs.sort(key=lambda spec: (spec["scale"], spec["uh"] * spec["uw"]), reverse=True)
    for spec in patch_specs:
        marker = np.full((spec["uh"], spec["uw"]), 1, dtype=int)
        try:
            ur, uc = random_free_location(
                reserve,
                marker,
                transparent=bg,
                border=PLACEMENT_BORDER,
                padding=PLACEMENT_PADDING,
                padding_connectivity=PLACEMENT_CONNECTIVITY,
            )
        except ValueError:
            return None

        paste(reserve, marker, ur, uc)
        paste(out, spec["output_patch"], ur + spec["out_r"], uc + spec["out_c"], transparent=bg)
        paste(inp, spec["input_patch"], ur + spec["in_r"], uc + spec["in_c"], transparent=bg)

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid c7f57c3e D2 input")

# Combo variant of c7f57c3e (index 29)
# combo_spec: D2/1+D15/2
# Name: 按边缘颜色配对 + 边缘色带编码规则

import random
from collections import defaultdict

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis
# ===============
# fused_variant:
#   input_concretization:
#     - 每个 sprite 仍是可按整数倍 NN 放大的多色精灵，内部保留 color-1 anchor。
#     - 与原题不同，最小原型从 2 个变成 4 个：2 个 edge-color 家族 x 2 个结构组(A/B)。
#     - 每个原型最上边都有一整行统一颜色的外边缘标签；该颜色既是 D2 的配对键，
#       也是 D15 色带中的“源色”。
#     - 网格上额外有一条 top/left 边缘色带，按顺序编码
#       [edge_src_0, edge_tgt_0, edge_src_1, edge_tgt_1]。
#     - 精灵内部两个可变角色只使用两种源色，而且这两种源色恰好就是两种 edge 标签色，
#       所以“边缘颜色”在对象边界、对象内部、边缘图例里三处重复出现。
#   behavior_trace:
#     - 原题：恢复 2 个 prototype -> 判断当前属于哪一个 -> 替换成另一个同尺度 prototype。
#     - D2 单变体：恢复 4 个 tagged prototype -> 先按 top edge color 选同家族 -> 再换到家族内另一结构。
#     - D15 单变体：恢复 2 个结构 prototype -> 读图例得到 src->tgt 映射 -> 结构互换后重着色。
#     - 本融合：先剥离色带，再恢复 4 个 tagged prototype；每个 sprite 先用自己的 edge color
#       找到“同标签家族里的另一结构”，随后把该替换原型中出现的 edge 源色整体映射到图例给出的
#       target 色。示例：若图例是 [6,2,8,4]，则 6/8 两种边缘色同时承担两件事：
#       `6` 的 top-row 精灵只能与另一个 `6` 家族原型互换结构，同时该输出精灵内所有 `6`
#       都要变成 `2`；`8` 同理变成 `4`。
#   complexity_calibration:
#     - 只保留 2 个 edge 家族、2 个 target 色、1 条图例，避免自由度过多。
#     - 每个样例固定出现全部 4 个 scale-1 原型，再额外放 1 个放大体，保证人类能同时看见：
#       家族内 A/B 配对关系、两种 edge 标签的重复、以及图例颜色顺序。
#     - 输出色不出现在输入 sprite 区，只出现在图例中，因此不读色带无法猜对颜色；
#       家族间 edge 标签都重复出现，因此不看边缘配对又会选错结构来源。
#   implicit_assumptions_review:
#     - 原版“所有非 bg 连通块都是 sprite”失效，因为色带本身也是前景；solver 必须先剥离图例。
#     - 原版 `len(uniq) == 2` 失效；融合后最小面积层必须恢复 4 个不同 tagged prototype。
#     - 原版 `repl_idx = 1 - best_idx` 失效；现在要先按 edge 标签分桶，再在桶内取另一个。
#     - 原版可用 overlap score 在 2 个原型间猜类型；融合后 4 个原型共存，且图例颜色参与重着色，
#       直接猜分会把家族与结构混淆，因此改为“整数尺度 + 精确匹配缩放原型”。
#     - 原版默认目标颜色已在最小样本中出现；融合后 target 色只在图例中出现，所以必须显式重着色。
#     - 新冲突审查：若图例源色和 edge 标签色不是同一组颜色，则“双重功能”会断裂；
#       generator 因此强制 `mapping.keys() == edge_families.keys()`。
#   placement_legality:
#     - generator 继续使用 output-first，并对每个 input/output patch 计算 anchor 对齐后的 union box，
#       再用 `random_free_location` 统一预留空间，保证输入输出都不越界且不会互相覆盖。
#     - 图例所在边缘旁额外预留一整行/列空白隔离带，避免图例与任意 sprite 发生 8 连通粘连。
#   visual_observability:
#     - top row 的统一 edge 色非常醒目，人眼容易把它识别为标签。
#     - 图例中的奇数位颜色与精灵边缘标签完全相同，偶数位颜色只出现在输出中，规则关联清晰。
#     - 4 个最小原型让“同标签成对 + 色带映射”都在单个样例内可见，不需要依赖隐藏排序。
#   visual_tidiness:
#     - 图例独占边缘单行/单列，和 sprite 区之间留空；对象之间维持 padding，避免视觉混杂。
#     - 边缘标签只占最上方一行，不与 anchor 或孔洞叠加。
#   color_inference:
#     - 颜色变化严格限制为两条映射，且源色就是两种反复出现的 edge 标签色。
#     - 颜色不按数值大小编码，完全由图例顺序决定；target 色与 src 色两两不同，避免链式覆盖。
#   new_main_assumptions:
#     - 恰有一条 top/left 图例，且其奇数位颜色正好等于 2 个 edge 家族标签色。
#     - 最小面积层恰有 4 个不同 prototype，且每个 edge 家族恰好 2 个原型。
#     - 所有 sprite 都是这 4 个原型的整数倍缩放拷贝；color-1 anchor bbox 在 4 个原型中同尺寸。
#     - 每个原型内部同时含有两种源色，因此图例映射对每个输出 sprite 都是可见且必要的。


def _pick_anchor_bbox(arr: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    """Bounding box of color-1 anchor cells."""
    if np.any(arr == 1):
        ys, xs = np.where(arr == 1)
        return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())
    raise AssertionError("sprite array has no color-1 pixels")


def _scale_nn(arr: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def _compress_runs(values: np.ndarray) -> list[int]:
    runs: list[int] = []
    for v in map(int, values.tolist()):
        if not runs or runs[-1] != v:
            runs.append(v)
    return runs


def _decode_instruction_band(grid: np.ndarray, bg: int) -> tuple[np.ndarray, dict[int, int]]:
    """Strip the edge legend and decode the two source->target mappings."""
    g = grid.copy()
    if np.any(g[0, 1:] != bg):
        pos = np.where(g[0] != bg)[0]
        assert pos.size > 0, "top legend must contain colored cells"
        assert np.all(np.diff(pos) == 1), "top legend must be contiguous"
        band_colors = _compress_runs(g[0, pos])
        g[0, :] = bg
    elif np.any(g[1:, 0] != bg):
        pos = np.where(g[:, 0] != bg)[0]
        assert pos.size > 0, "left legend must contain colored cells"
        assert np.all(np.diff(pos) == 1), "left legend must be contiguous"
        band_colors = _compress_runs(g[pos, 0])
        g[:, 0] = bg
    else:
        raise AssertionError("missing instruction band")

    assert len(band_colors) == 4, "instruction band must encode exactly two mappings"
    assert len(set(band_colors)) == 4, "band colors must be pairwise distinct"
    mapping = {
        int(band_colors[0]): int(band_colors[1]),
        int(band_colors[2]): int(band_colors[3]),
    }
    return g, mapping


def _edge_color(arr: np.ndarray, bg: int) -> int:
    """Read the uniform top-edge tag color from a tagged prototype/sprite."""
    top = arr[0]
    col = int(top[0])
    assert col not in (int(bg), 1), "edge tag must be non-bg and non-anchor"
    assert np.all(top == col), "edge tag row must be uniform"
    return col


def _infer_scale(arr: np.ndarray, base_anchor_h: int, base_anchor_w: int, bg: int) -> int:
    ar0, ac0, ar1, ac1 = _pick_anchor_bbox(arr, bg)
    inst_h = int(ar1 - ar0 + 1)
    inst_w = int(ac1 - ac0 + 1)
    assert inst_h % int(base_anchor_h) == 0 and inst_w % int(base_anchor_w) == 0
    scale_h = inst_h // int(base_anchor_h)
    scale_w = inst_w // int(base_anchor_w)
    assert scale_h == scale_w and scale_h >= 1
    return int(scale_h)


def _match_scaled_proto(arr: np.ndarray, protos: list[np.ndarray], scale: int) -> int:
    for idx, proto in enumerate(protos):
        cand = _scale_nn(proto, scale)
        if cand.shape == arr.shape and np.array_equal(cand, arr):
            return int(idx)
    raise AssertionError("sprite does not match any scaled prototype")


def _recolor_proto(arr: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = arr.copy()
    for src, dst in mapping.items():
        out[arr == src] = dst
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap each sprite within its edge-color family, then recolor it using the edge legend.

    Input structure:
    - The dominant background surrounds one top-row or left-column legend band plus 5 sprite components.
    - The smallest sprites reveal 4 tagged prototypes: 2 edge-color families, each family containing two opposite structures.
    - Every sprite is an integer NN-scaled copy of one tagged prototype, and the two edge-tag colors also serve as the only non-anchor source colors used inside the sprites.

    Rule:
    - Read the edge legend in order as edge_src_0, edge_tgt_0, edge_src_1, edge_tgt_1, then remove that legend from consideration.
    - Recover the 4 smallest tagged prototypes, group them by their uniform top-edge color, and identify each sprite's exact scaled prototype.
    - Replace each sprite with the other prototype from the same edge-color family at the same scale and anchor alignment, then recolor all source edge colors according to the legend mapping.

    Key insight: The same two edge colors play two roles at once: a sprite's outer tag chooses which family supplies the replacement structure, and the edge legend tells how those very same source colors must be recolored in the output.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    work, color_map = _decode_instruction_band(g, bg)

    comps = components_mask(work != bg, connectivity=8)
    assert len(comps) >= 5, "grid must contain at least 5 sprite components after removing the legend"

    sprites = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = work[r0 : r1 + 1, c0 : c1 + 1].copy()
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
    assert all(len(idxs) == 2 for idxs in edge_buckets.values()), "each edge family must contain 2 prototypes"
    assert set(edge_buckets.keys()) == set(color_map.keys()), "legend source colors must equal the edge-family colors"

    anc0 = _pick_anchor_bbox(protos[0], bg)
    base_anchor_h = int(anc0[2] - anc0[0] + 1)
    base_anchor_w = int(anc0[3] - anc0[1] + 1)

    out = g.copy()

    for s in sprites:
        arr = s["arr"]
        scale = _infer_scale(arr, base_anchor_h, base_anchor_w, bg)
        cur_idx = _match_scaled_proto(arr, protos, scale)

        family = _edge_color(protos[cur_idx], bg)
        mates = edge_buckets[family]
        repl_idx = int(mates[0] if mates[1] == cur_idx else mates[1])

        for rr, cc in s["pts"]:
            out[int(rr), int(cc)] = bg

        r0, c0, _r1, _c1 = s["bbox"]
        ar0, ac0, _ar1, _ac1 = _pick_anchor_bbox(arr, bg)
        repl_src = protos[repl_idx]
        repl_out = _recolor_proto(repl_src, color_map)
        scaled_out = _scale_nn(repl_out, scale)
        pr0, pc0, _pr1, _pc1 = _pick_anchor_bbox(repl_src, bg)
        top = int(r0 + ar0 - pr0 * scale)
        left = int(c0 + ac0 - pc0 * scale)
        paste(out, scaled_out, top, left, transparent=bg)

    return out


# --------------- generator (o5x) ---------------

# --- CFG: canvas ---
GRID_SIZE_MIN = 22
GRID_SIZE_MAX = 32
GRID_MARGIN_FOR_SPRITES = 9

# --- CFG: prototypes ---
CORE_SHAPES = [(3, 3), (3, 4), (4, 4)]
BG_CHOICES = [0, 2, 3, 4, 5, 6, 7, 8, 9]

# --- CFG: row-mask weights ---
ANCHOR_ROW_WEIGHTS = [("center", 0.45), ("outer", 0.35), ("full", 0.20)]
VARIABLE_ROW_WEIGHTS = [("center", 0.68), ("full", 0.32)]
OUTER_FIXUP_PROB = 0.7

# --- CFG: sprite instances ---
BIG_SCALES = [2, 3]

# --- CFG: instruction band ---
BAND_ORIENTATIONS = ["top", "left"]
BAND_BLOCK_CHOICES = [2, 3]

# --- CFG: placement ---
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2
PLACEMENT_CONNECTIVITY = 8

# --- CFG: generation retries ---
MAX_RETRIES = 500

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
    """Build a core label grid with roles 1, 2, 3 and bg holes 0."""
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
        masks.append(_row_mask(random.choices(kinds, weights=wts)[0], w))

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
    assert np.any(labels == 2)
    assert np.any(labels == 3)
    return labels


def _colorize_core(labels: np.ndarray, colors: tuple[int, int], bg: int) -> np.ndarray:
    core = np.full(labels.shape, bg, dtype=int)
    core[labels == 1] = 1
    core[labels == 2] = int(colors[0])
    core[labels == 3] = int(colors[1])
    return core


def _tagged_proto(core: np.ndarray, edge_color: int) -> np.ndarray:
    edge = np.full((1, core.shape[1]), int(edge_color), dtype=int)
    return np.vstack([edge, core])


def _make_prototypes(
    bg: int,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[int, int], tuple[int, int, int, int]]:
    """Create 4 source prototypes and their recolored target versions."""
    for _ in range(60):
        labels = _make_labels()
        flipped = np.flipud(labels)
        if np.array_equal(labels, flipped):
            continue

        pool = [c for c in range(10) if c not in (bg, 1)]
        edge_colors = random.sample(pool, 2)
        remaining = [c for c in pool if c not in edge_colors]
        target_colors = random.sample(remaining, 2)
        mapping = {int(edge_colors[0]): int(target_colors[0]), int(edge_colors[1]): int(target_colors[1])}

        proto_a0 = _tagged_proto(_colorize_core(labels, (edge_colors[0], edge_colors[1]), bg), edge_colors[0])
        proto_b0 = _tagged_proto(_colorize_core(flipped, (edge_colors[0], edge_colors[1]), bg), edge_colors[0])
        proto_a1 = _tagged_proto(_colorize_core(labels, (edge_colors[1], edge_colors[0]), bg), edge_colors[1])
        proto_b1 = _tagged_proto(_colorize_core(flipped, (edge_colors[1], edge_colors[0]), bg), edge_colors[1])

        src_protos = [proto_a0, proto_b0, proto_a1, proto_b1]
        if len({tuple(p.ravel().tolist()) for p in src_protos}) != 4:
            continue

        pair_map = {0: 1, 1: 0, 2: 3, 3: 2}
        tgt_protos = [_recolor_proto(p, mapping) for p in src_protos]
        band_colors = (int(edge_colors[0]), int(target_colors[0]), int(edge_colors[1]), int(target_colors[1]))

        assert all(set(np.unique(p)) >= {1, edge_colors[0], edge_colors[1]} for p in src_protos)
        return src_protos, tgt_protos, pair_map, band_colors

    raise RuntimeError("failed to make asymmetric tagged prototypes")


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


def _reserve_instruction_lane(reserve: np.ndarray, orientation: str) -> None:
    if orientation == "top":
        reserve[0:2, :] = 1
    else:
        reserve[:, 0:2] = 1


def _paint_instruction_band(
    canvas: np.ndarray,
    orientation: str,
    band_colors: tuple[int, int, int, int],
    block: int,
    start: int | None = None,
) -> int:
    n = canvas.shape[0]
    seq = [color for color in band_colors for _ in range(block)]
    band_len = len(seq)
    if start is None:
        start = random.randint(1, n - band_len - 1)
    if orientation == "top":
        canvas[0, start : start + band_len] = seq
    else:
        canvas[start : start + band_len, 0] = seq
    return start


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_CHOICES)
    src_protos, tgt_protos, pair_map, band_colors = _make_prototypes(bg)

    big_scale = random.choice(BIG_SCALES)
    instances = [(idx, 1) for idx in range(4)] + [(random.randrange(4), big_scale)]

    patch_specs = []
    max_union_dim = 0
    for input_idx, scale in instances:
        input_patch = _scale(src_protos[input_idx], scale)
        output_patch = _scale(tgt_protos[pair_map[input_idx]], scale)
        out_r, out_c, in_r, in_c, uh, uw = _aligned_union_offsets(output_patch, input_patch)
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

    orientation = random.choice(BAND_ORIENTATIONS)
    block = random.choice(BAND_BLOCK_CHOICES)
    band_span = 4 * block + 2

    low = max(GRID_SIZE_MIN, max_union_dim + GRID_MARGIN_FOR_SPRITES, band_span)
    if low > GRID_SIZE_MAX:
        return None
    n = random.randint(low, GRID_SIZE_MAX)

    inp = np.full((n, n), bg, dtype=int)
    out = np.full((n, n), bg, dtype=int)
    reserve = np.full((n, n), bg, dtype=int)

    _reserve_instruction_lane(reserve, orientation)
    band_start = _paint_instruction_band(inp, orientation, band_colors, block)
    _paint_instruction_band(out, orientation, band_colors, block, start=band_start)

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
    raise RuntimeError("failed to generate a valid c7f57c3e combo input")

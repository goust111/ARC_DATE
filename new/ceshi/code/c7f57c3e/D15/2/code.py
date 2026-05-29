# Variant of c7f57c3e
# Dimension: D15 — 符号编码/图例 (symbol_instructions)
# Change: 无符号编码，颜色模式直接从精灵颜色推断分组... → 边缘色带编码交换规则

# Design analysis
# ===============
# spec:
#   preserved_core:
#     - 仍然只有两种结构原型；每个精灵仍然按 color-1 锚框对齐、按原尺度替换成另一种结构。
#     - 精灵数量、尺度分布、放置间距、output-first 反推方式都尽量沿用原题。
#   new_symbol_instruction:
#     - 输入的最上边一行或最左边一列放置一条四色带，颜色顺序编码两个映射：
#       第 1 色 -> 第 2 色，第 3 色 -> 第 4 色。
#     - 色带用长度 2 或 3 的纯色块重复显示每个颜色，保证肉眼可见。
#     - 这条色带在输入和输出中都保留不变，只承担“图例/说明书”作用。
#   sprite_colors:
#     - 输入区的两种结构原型共享同一组源颜色 src1/src2；因此仅看精灵本体，能知道结构翻转，
#       但不知道输出应该换成什么新颜色。
#     - 输出区把替换后的精灵重着色为 tgt1/tgt2，其中 tgt1/tgt2 只在色带中提前出现。
#
# behavior_trace:
#   original_local_step:
#     - 最小精灵直接给出 proto_A / proto_B 的完整颜色模式；solver 仅靠这些最小样本即可复原输出。
#   variant_local_step:
#     - 最小精灵仍然给出两种结构，但两者都只使用 src1/src2。
#     - solver 先从边缘色带读出 [src1, tgt1, src2, tgt2]，再把“另一种结构原型”按该映射重着色后贴回。
#   concrete_example:
#     - 若色带为 [6,2,8,4]，则映射是 6->2, 8->4。
#     - 一个输入 A 型精灵内部用 {1,6,8}，输出时先切到 B 型结构，再把 6/8 改成 2/4。
#     - 不读色带就只能得到“切结构”，得不到正确的新颜色，因此不会退化回原题。
#
# complexity_calibration:
#   - 只使用两个颜色映射对，避免颜色自由度过大。
#   - src/tgt 四种颜色强制互异，防止链式覆盖和人眼歧义。
#   - 色带方向只在 top / left 二选一，块长只在 2 / 3 二选一，样例中会重复出现这些离散档位。
#
# implicit_assumption_audit_of_original_main:
#   - 原版把全部非背景连通块都当作精灵；这里不成立，因为色带也是非背景对象，所以必须先剥离边缘图例。
#   - 原版默认最小精灵的颜色模式已经包含“目标颜色”；这里不成立，因为目标颜色只出现在色带里。
#   - 原版直接把另一原型数组贴回不会出错；这里若仍这样做，就只能得到源颜色而不是目标颜色。
#   - 原版对颜色值本身没有顺序依赖；变体继续保持这一点，颜色对完全按可见的色带次序解码，绝不按 id 排序。
#
# placement_legality:
#   - generator 在带图例的边缘旁额外保留一整行/列空白隔离带，避免色带与任意精灵 8 连通。
#   - 精灵依旧用 reserve + random_free_location 放置，保持不重叠、不越界、bbox 间至少 2 格。
#   - 输出与输入共用同一条色带，所以 solver 清空和回贴时只处理 sprite components，不改动图例。
#
# visual_observability:
#   - 结构翻转仍然可见；同时输出中新颜色在输入精灵区完全缺席，只能从边缘色带推出。
#   - 色带是单独的线性对象，与主体精灵区分明显，不会和锚点或孔洞混在一起。
#   - 目标颜色数量固定为 2，人类可在少量样例中观察到“奇数位色块对应偶数位色块”的规律。
#
# new_main_assumptions_and_generator_guarantees:
#   - 假设恰有一条边缘色带，并且 top 色带不占用第 0 列、left 色带不占用第 0 行；generator 明确保证。
#   - 假设最小面积精灵中能提取出两种不同结构；generator 避免使用上下翻转后完全相同的标签布局。
#   - 假设所有精灵共享同一对源颜色，且 color-1 锚框在两原型中的尺寸相同；generator 统一构造并验证。

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


def _pick_anchor_bbox(arr: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    """Pick a stable anchor bbox inside the sprite array (color-1 pixels; always present)."""
    if np.any(arr == 1):
        ys, xs = np.where(arr == 1)
        return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())
    assert False, "sprite array has no color-1 pixels and should always have them"


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
    """Strip the top/left legend band and return the source->target color mapping."""
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
        assert False, "missing instruction band"

    assert len(band_colors) == 4, "instruction band must encode exactly two color mappings"
    assert len(set(band_colors)) == 4, "instruction band colors must be distinct"
    mapping = {
        int(band_colors[0]): int(band_colors[1]),
        int(band_colors[2]): int(band_colors[3]),
    }
    return g, mapping


def _recolor_proto(arr: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = arr.copy()
    for src, dst in mapping.items():
        out[arr == src] = dst
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap each sprite to the opposite structural prototype, with recoloring dictated by an edge legend.

    Input structure:
    - The background is the dominant color, and one edge contains a short top-row or left-column color band.
    - Away from that band sit 3-4 scaled sprite instances of two structural prototypes; both input prototypes share the same two non-anchor source colors.

    Rule:
    - Read the band in order as source1, target1, source2, target2, which defines two color remaps.
    - Recover the two structural prototypes from the smallest sprites, determine each sprite's type and integer scale, and replace it with the opposite structure at the same scale.
    - Recolor the replacement by applying source1->target1 and source2->target2, while keeping the color-1 anchor region and the legend band unchanged.

    Key insight: The sprite field exposes only the structural toggle and the source palette; the output palette exists only in the edge band, so solving requires decoding that legend before recoloring the swapped prototype.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    work, color_map = _decode_instruction_band(g, bg)

    comps = components_mask(work != bg, connectivity=8)
    assert len(comps) >= 2, "grid must contain at least 2 sprite components"

    sprites = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = work[r0 : r1 + 1, c0 : c1 + 1].copy()
        sprites.append({"pts": pts, "bbox": (r0, c0, r1, c1), "arr": arr})

    areas = [int((s["bbox"][2] - s["bbox"][0] + 1) * (s["bbox"][3] - s["bbox"][1] + 1)) for s in sprites]
    min_area = min(areas)
    base = [s for s, a in zip(sprites, areas, strict=False) if a == min_area]
    uniq: dict[tuple[int, ...], np.ndarray] = {}
    for s in base:
        key = tuple(s["arr"].ravel().tolist())
        if key not in uniq:
            uniq[key] = s["arr"]
    assert len(uniq) == 2, "smallest-area sprites must yield exactly 2 distinct structural prototypes"

    proto_keys = list(uniq.keys())[:2]
    protos = [uniq[proto_keys[0]], uniq[proto_keys[1]]]
    recolored = [_recolor_proto(p, color_map) for p in protos]

    anc0 = _pick_anchor_bbox(protos[0], bg)
    a0r0, a0c0, a0r1, a0c1 = anc0
    base_anc_h = a0r1 - a0r0 + 1
    base_anc_w = a0c1 - a0c0 + 1

    out = g.copy()

    for s in sprites:
        r0, c0, _r1, _c1 = s["bbox"]
        arr = s["arr"]
        anc = _pick_anchor_bbox(arr, bg)
        ar0, ac0, ar1, ac1 = anc
        inst_anc_h = ar1 - ar0 + 1
        inst_anc_w = ac1 - ac0 + 1
        scale_h = max(1, inst_anc_h // max(1, base_anc_h))
        scale_w = max(1, inst_anc_w // max(1, base_anc_w))
        scale = int(max(scale_h, scale_w))

        best_idx = 0
        best_score = -1
        for idx, p in enumerate(protos):
            sp = _scale_nn(p, scale)
            panc = _pick_anchor_bbox(p, bg)
            pr0, pc0, _pr1, _pc1 = panc
            pr0 *= scale
            pc0 *= scale
            top = (r0 + ar0) - pr0
            left = (c0 + ac0) - pc0

            score = 0
            ph, pw = sp.shape
            rr0 = max(0, top)
            cc0 = max(0, left)
            rr1 = min(h, top + ph)
            cc1 = min(w, left + pw)
            if rr0 >= rr1 or cc0 >= cc1:
                continue
            sub_g = work[rr0:rr1, cc0:cc1]
            sub_p = sp[(rr0 - top) : (rr1 - top), (cc0 - left) : (cc1 - left)]
            m = sub_p != bg
            if np.any(m):
                score = int(np.sum(sub_g[m] == sub_p[m]))
            if score > best_score:
                best_score = score
                best_idx = idx

        repl_idx = 1 - best_idx

        for rr, cc in s["pts"]:
            out[rr, cc] = bg

        p = recolored[repl_idx]
        sp = _scale_nn(p, scale)
        panc = _pick_anchor_bbox(protos[repl_idx], bg)
        pr0, pc0, _pr1, _pc1 = panc
        pr0 *= scale
        pc0 *= scale
        top = (r0 + ar0) - pr0
        left = (c0 + ac0) - pc0

        ph, pw = sp.shape
        rr0 = max(0, top)
        cc0 = max(0, left)
        rr1 = min(h, top + ph)
        cc1 = min(w, left + pw)
        sub_p = sp[(rr0 - top) : (rr1 - top), (cc0 - left) : (cc1 - left)]
        m = sub_p != bg
        out_slice = out[rr0:rr1, cc0:cc1]
        out_slice[m] = sub_p[m]
        out[rr0:rr1, cc0:cc1] = out_slice

    return out


# --------------- generator (o5x) ---------------

# --- CFG: canvas ---
GRID_SIZE_MIN = 18
GRID_SIZE_MAX = 28
GRID_MARGIN_FOR_SPRITES = 6

# --- CFG: prototypes ---
PROTO_SHAPES = [(4, 3), (3, 4), (4, 4)]
BG_CHOICES = [0, 2, 3, 4, 5, 6, 7, 8, 9]

# --- CFG: row-mask weights (anchor role vs variable role) ---
ANCHOR_ROW_WEIGHTS = [("center", 0.45), ("outer", 0.35), ("full", 0.20)]
VARIABLE_ROW_WEIGHTS = [("center", 0.68), ("full", 0.32)]
OUTER_FIXUP_PROB = 0.7

# --- CFG: sprite instances ---
SPRITE_COUNTS = [3, 4]
BIG_SCALES = [2, 3]

# --- CFG: instruction band ---
BAND_ORIENTATIONS = ["top", "left"]
BAND_BLOCK_CHOICES = [2, 3]

# --- CFG: placement ---
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2
PLACEMENT_CONNECTIVITY = 8

# --- CFG: generation retries ---
MAX_RETRIES = 300

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
    h, w = random.choice(PROTO_SHAPES)

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
        entries = weights_by_role[role]
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
        masks[random.randrange(h)] = _row_mask("center", w)

    labels = np.zeros((h, w), dtype=int)
    for r, (role, mask) in enumerate(zip(roles, masks)):
        for c, bit in enumerate(mask):
            if bit:
                labels[r, c] = role
    return labels


def _make_prototypes(
    bg: int,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[int, int, int, int]]:
    for _ in range(40):
        labels = _make_labels()
        if np.array_equal(labels, np.flipud(labels)):
            continue

        pool = [c for c in range(10) if c not in (bg, 1)]
        band_colors = random.sample(pool, 4)
        src_cols = [band_colors[0], band_colors[2]]
        tgt_cols = [band_colors[1], band_colors[3]]

        def _colorize(lab: np.ndarray, cols: list[int]) -> np.ndarray:
            proto = np.full(lab.shape, bg, dtype=int)
            proto[lab == 1] = 1
            proto[lab == 2] = cols[0]
            proto[lab == 3] = cols[1]
            return proto

        proto_a_src = _colorize(labels, src_cols)
        proto_b_src = _colorize(np.flipud(labels), src_cols)
        if np.array_equal(proto_a_src, proto_b_src):
            continue
        proto_a_tgt = _colorize(labels, tgt_cols)
        proto_b_tgt = _colorize(np.flipud(labels), tgt_cols)
        return (proto_a_src, proto_b_src), (proto_a_tgt, proto_b_tgt), tuple(band_colors)
    raise RuntimeError("failed to make asymmetric prototypes")


def _union_bbox(patch_a: np.ndarray, patch_b: np.ndarray) -> tuple[int, int, int, int]:
    oa = _anchor_bbox(patch_a)
    ia = _anchor_bbox(patch_b)
    dr = oa[0] - ia[0]
    dc = oa[1] - ia[1]
    h, w = patch_a.shape
    return dr, dc, h + abs(dr), w + abs(dc)


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
    src_protos, tgt_protos, band_colors = _make_prototypes(bg)

    count = random.choice(SPRITE_COUNTS)
    big_scale = random.choice(BIG_SCALES)
    instances = [(0, 1), (1, 1), (random.randrange(2), big_scale)]
    if count == 4:
        instances.append((random.randrange(2), 1))
    random.shuffle(instances)
    instances.sort(key=lambda item: item[1], reverse=True)

    orientation = random.choice(BAND_ORIENTATIONS)
    block = random.choice(BAND_BLOCK_CHOICES)
    band_span = 4 * block + 2

    max_dim = max(max(src_protos[typ].shape) * scale for typ, scale in instances)
    n = random.randint(max(GRID_SIZE_MIN, max_dim + GRID_MARGIN_FOR_SPRITES, band_span), GRID_SIZE_MAX)
    out = np.full((n, n), bg, dtype=int)
    inp = np.full((n, n), bg, dtype=int)
    reserve = np.full((n, n), bg, dtype=int)

    _reserve_instruction_lane(reserve, orientation)
    band_start = _paint_instruction_band(inp, orientation, band_colors, block)
    _paint_instruction_band(out, orientation, band_colors, block, start=band_start)

    for out_type, scale in instances:
        output_patch = _scale(tgt_protos[out_type], scale)
        input_patch = _scale(src_protos[1 - out_type], scale)
        dr, dc, uh, uw = _union_bbox(output_patch, input_patch)
        marker = np.full((uh, uw), 1, dtype=int)
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

        out_r = ur + max(0, -dr)
        out_c = uc + max(0, -dc)
        in_r, in_c = out_r + dr, out_c + dc

        paste(reserve, marker, ur, uc)
        paste(out, output_patch, out_r, out_c, transparent=bg)
        paste(inp, input_patch, in_r, in_c, transparent=bg)

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
    raise RuntimeError("failed to generate a valid c7f57c3e input")

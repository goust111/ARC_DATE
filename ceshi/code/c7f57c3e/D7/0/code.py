# Variant of c7f57c3e
# Dimension: D7 — 尺度保持规则 (meta)
# Change: 交换颜色模式时保持精灵的尺度（scale）不变，同尺度精灵配对... → 交换时同时缩放到目标尺度
#
# Design analysis
# spec:
#   core_change:
#     original: |
#       原题中每个位置只会切换到“另一种原型”，尺度仍取当前位置自己的尺度；
#       因为组内所有实例都是同一原型的整数缩放副本，所以即使允许“异尺度配对”，
#       如果最后仍把对方图案缩放回当前位置自己的尺度，输出会退化成原题，肉眼看不出 D7 的变化。
#     variant: |
#       因此这里把 D7 具体化为“成对交换原型时，尺度也随源精灵一起跨位置交换”：
#       第 k 个 A 组精灵与第 k 个 B 组精灵配对后，
#       A 的位置接收 B 组图案和 B 的尺度，B 的位置接收 A 组图案和 A 的尺度。
#       仍然沿用 color-1 锚框对齐，所以位置语义保留，但每个位置上的尺寸会改变。
#   visible_layout:
#     input: |
#       画面里始终有 4 个 8 连通精灵，恰好分成两组（两种原型）。
#       两组各有 2 个实例，并按从上到下的可见顺序配对。
#       每一行都有一对左右相对的精灵：一边是 scale=1，另一边是 scale=2。
#     output: |
#       同一行的两个精灵交换到对方的原型和尺度，锚点仍落在原来那一行各自的位置上；
#       所以视觉上会看到“小变大、大变小”的互换，而不是原题那种“只换花纹不换尺寸”。
#   small_trace:
#     before_after: |
#       例：上排左侧是 A@1，右侧是 B@2；下排左侧是 A@2，右侧是 B@1。
#       原题输出会变成左 B@1 / 右 A@2 和 左 B@2 / 右 A@1。
#       本变体输出改成左 B@2 / 右 A@1 和 左 B@1 / 右 A@2。
#       差异正好落在“输出位置采用谁的尺度”。
#   complexity_check:
#     note: |
#       只保留两档尺度（1 和 2）。这样足够让尺寸交换可见，同时避免 2x2 布局在 18-28 网格里过于拥挤。
#       每张图里有两对、且两对都发生异尺度交换，避免偶然退化。
#   hidden_assumptions_review:
#     original_solver_dependencies:
#       - "原代码用最小面积实例恢复两个基准原型；这里继续保证每组各有一个 scale=1 实例，所以该前提仍成立。"
#       - "原代码逐个位置独立切换原型；这依赖'目标尺度就是当前位置尺度'。变体不再成立，必须先分组再跨组配对。"
#       - "原代码输出可在原 bbox 附近覆盖；变体中输出 bbox 可能因拿到对方尺度而变大，所以生成器必须为每个位置预留 input/output 联合包围盒。"
#     new_solver_dependencies:
#       - "两组实例数量必须相等；生成器固定为 2 对。"
#       - "配对顺序必须可见且稳定；生成器保证每组按锚点从上到下排序后正好对应两行。"
#       - "所有实例都是基准原型的整数最近邻缩放；这样可以通过锚框尺寸恢复 scale，并用降采样恢复逻辑原型。"
#   legality_and_cleanliness:
#     placement: |
#       生成器不用全局自由散布，而是把 4 个位置放进 2x2 单元格。
#       每个单元格的尺寸由该位置 input/output 两个补丁的联合包围盒决定，
#       再加固定边框与间距，因此不会越界、不会重叠，也不会因为输出变大而碰撞。
#     observability: |
#       配对依据只用可见空间顺序，不用颜色 id 或隐藏排序。
#       每个样例中两对都发生“小↔大”交换，因此 2-3 组样例足以看出“尺度会随配对源一起交换”。

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


def _pick_anchor_bbox(arr: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    """Return the bbox of color-1 anchor pixels."""
    ys, xs = np.where(arr == 1)
    assert ys.size > 0, "every sprite must contain color 1 anchor pixels"
    return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())


def _scale_nn(arr: np.ndarray, scale: int) -> np.ndarray:
    """Nearest-neighbor upscale by an integer factor."""
    scale = int(scale)
    if scale <= 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def _descale_nn(arr: np.ndarray, scale: int) -> np.ndarray:
    """Recover the logical prototype from an exact NN-scaled sprite."""
    scale = int(scale)
    if scale <= 1:
        return arr.copy()
    h, w = arr.shape
    assert h % scale == 0 and w % scale == 0, "sprite shape must be divisible by scale"
    return arr[::scale, ::scale].copy()


def _aligned_union_bbox(
    output_patch: np.ndarray, input_patch: np.ndarray, bg: int
) -> tuple[int, int, int, int]:
    """Union box for anchor-aligned output/input placement at one logical slot."""
    oa = _pick_anchor_bbox(output_patch, bg)
    ia = _pick_anchor_bbox(input_patch, bg)
    dr = oa[0] - ia[0]
    dc = oa[1] - ia[1]

    oh, ow = output_patch.shape
    ih, iw = input_patch.shape
    union_h = max(oh - min(0, dr), ih + max(0, dr))
    union_w = max(ow - min(0, dc), iw + max(0, dc))
    return int(dr), int(dc), int(union_h), int(union_w)


def _place_from_union_origin(
    union_r: int, union_c: int, dr: int, dc: int
) -> tuple[int, int, int, int]:
    """Convert union origin + relative anchor offset into output/input top-lefts."""
    out_r = union_r - min(0, dr)
    out_c = union_c - min(0, dc)
    in_r = out_r + dr
    in_c = out_c + dc
    return int(out_r), int(out_c), int(in_r), int(in_c)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap paired sprites to the opposite prototype while also exchanging their scales.

    Input structure:
    - The background is the most common color. Exactly two multicolor sprite prototypes appear as connected components.
    - Each prototype occurs twice, once at scale 1 and once at scale 2, and every sprite contains a color-1 anchor region.

    Rule:
    - Recover the two base prototypes from the smallest sprites, then classify every sprite by prototype and scale.
    - Sort the sprites within each prototype by anchor position, pair the k-th sprite of one group with the k-th sprite of the other group, and swap them across the pair.
    - When a sprite moves to its partner's location it keeps its own scale, so each location receives the opposite prototype at the partner sprite's scale, aligned by the target location's anchor.

    Key insight: Prototype identity and scale now travel together with the paired source sprite; the visible slot is fixed by the target anchor, but the size comes from the other group.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=8)
    assert len(comps) >= 2, "grid must contain at least two sprites"

    sprites = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        anc = _pick_anchor_bbox(arr, bg)
        sprites.append(
            {
                "pts": pts,
                "bbox": (r0, c0, r1, c1),
                "arr": arr,
                "anchor_bbox": anc,
                "anchor_global": (int(r0 + anc[0]), int(c0 + anc[1])),
            }
        )

    areas = [int((s["bbox"][2] - s["bbox"][0] + 1) * (s["bbox"][3] - s["bbox"][1] + 1)) for s in sprites]
    min_area = min(areas)
    base = [s for s, a in zip(sprites, areas, strict=False) if a == min_area]
    uniq: dict[tuple[int, ...], np.ndarray] = {}
    for s in base:
        key = tuple(s["arr"].ravel().tolist())
        if key not in uniq:
            uniq[key] = s["arr"]
    assert len(uniq) == 2, "the smallest sprites must give the two base prototypes"

    protos = list(uniq.values())[:2]
    anc0 = _pick_anchor_bbox(protos[0], bg)
    anc1 = _pick_anchor_bbox(protos[1], bg)
    base_anchor_h0 = anc0[2] - anc0[0] + 1
    base_anchor_w0 = anc0[3] - anc0[1] + 1
    base_anchor_h1 = anc1[2] - anc1[0] + 1
    base_anchor_w1 = anc1[3] - anc1[1] + 1
    assert (base_anchor_h0, base_anchor_w0) == (base_anchor_h1, base_anchor_w1)

    for s in sprites:
        ar0, ac0, ar1, ac1 = s["anchor_bbox"]
        inst_anchor_h = ar1 - ar0 + 1
        inst_anchor_w = ac1 - ac0 + 1
        assert inst_anchor_h % base_anchor_h0 == 0 and inst_anchor_w % base_anchor_w0 == 0
        scale_h = inst_anchor_h // base_anchor_h0
        scale_w = inst_anchor_w // base_anchor_w0
        assert scale_h == scale_w and scale_h >= 1
        scale = int(scale_h)

        logical = _descale_nn(s["arr"], scale)
        match_idx = None
        for idx, proto in enumerate(protos):
            if np.array_equal(logical, proto):
                match_idx = idx
                break
        assert match_idx is not None, "every sprite must descale to one of the two prototypes"
        s["type"] = int(match_idx)
        s["scale"] = scale

    groups = {0: [], 1: []}
    for s in sprites:
        groups[s["type"]].append(s)
    assert len(groups[0]) == len(groups[1]) >= 1, "the two prototype groups must have the same size"

    for idx in (0, 1):
        groups[idx].sort(key=lambda s: (s["anchor_global"][0], s["anchor_global"][1]))

    out = np.full_like(g, bg)

    for sa, sb in zip(groups[0], groups[1], strict=True):
        patch_for_a = _scale_nn(protos[sb["type"]], sb["scale"])
        patch_for_b = _scale_nn(protos[sa["type"]], sa["scale"])

        anc_patch_a = _pick_anchor_bbox(patch_for_a, bg)
        anc_patch_b = _pick_anchor_bbox(patch_for_b, bg)

        top_a = sa["anchor_global"][0] - anc_patch_a[0]
        left_a = sa["anchor_global"][1] - anc_patch_a[1]
        top_b = sb["anchor_global"][0] - anc_patch_b[0]
        left_b = sb["anchor_global"][1] - anc_patch_b[1]

        paste(out, patch_for_a, top_a, left_a, transparent=bg)
        paste(out, patch_for_b, top_b, left_b, transparent=bg)

    return out


# --------------- generator (o5x) ---------------

# --- CFG: canvas ---
GRID_SIZE_MIN = 18
GRID_SIZE_MAX = 28
GRID_BORDER = 1
ROW_GAP = 2
COL_GAP = 2

# --- CFG: prototypes ---
PROTO_SHAPES = [(4, 3), (3, 4), (4, 4)]
BG_CHOICES = [0, 2, 3, 4, 5, 6, 7, 8, 9]
COLOR_SHARE_PROB = 0.65

# --- CFG: row-mask weights ---
ANCHOR_ROW_WEIGHTS = [("center", 0.45), ("outer", 0.35), ("full", 0.20)]
VARIABLE_ROW_WEIGHTS = [("center", 0.68), ("full", 0.32)]
OUTER_FIXUP_PROB = 0.7

# --- CFG: variant scales ---
SMALL_SCALE = 1
BIG_SCALE = 2

# --- CFG: generation retries ---
MAX_RETRIES = 300

_ROW_MASKS = {
    "center": {3: [0, 1, 0], 4: [0, 1, 1, 0]},
    "outer": {3: [1, 0, 1], 4: [1, 0, 0, 1]},
    "full": {3: [1, 1, 1], 4: [1, 1, 1, 1]},
}


def _row_mask(kind: str, w: int) -> list[int]:
    return list(_ROW_MASKS[kind][w])


def _make_labels() -> np.ndarray:
    """Build a connected label grid with anchor role 1 and two variable roles."""
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
        kinds, wts = zip(*weights_by_role[role])
        masks.append(_row_mask(random.choices(kinds, weights=wts)[0], w))

    if not any(mask[0] or mask[-1] for mask in masks):
        idx = random.randrange(h)
        if roles[idx] == 1 and random.random() < OUTER_FIXUP_PROB:
            masks[idx] = _row_mask("outer", w)
        else:
            masks[idx] = _row_mask("full", w)

    if all(all(mask) for mask in masks):
        masks[random.randrange(h)] = _row_mask("center", w)

    labels = np.zeros((h, w), dtype=int)
    for r, (role, mask) in enumerate(zip(roles, masks, strict=False)):
        for c, bit in enumerate(mask):
            if bit:
                labels[r, c] = role

    comps = components_mask(labels != 0, connectivity=8)
    if len(comps) != 1:
        return _make_labels()
    return labels


def _make_prototypes(bg: int) -> tuple[np.ndarray, np.ndarray]:
    """Create proto_A and proto_B from one shared structural label grid."""
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


def _slot_record(
    target_type: int,
    target_scale: int,
    source_type: int,
    source_scale: int,
    protos: list[np.ndarray],
    bg: int,
) -> dict:
    """Describe one physical slot with its input/output patches and reserved union box."""
    output_patch = _scale_nn(protos[target_type], target_scale)
    input_patch = _scale_nn(protos[source_type], source_scale)
    dr, dc, uh, uw = _aligned_union_bbox(output_patch, input_patch, bg)
    return {
        "output_patch": output_patch,
        "input_patch": input_patch,
        "dr": dr,
        "dc": dc,
        "uh": uh,
        "uw": uw,
    }


def _gen_attempt() -> np.ndarray | None:
    """Construct one valid input by laying out two row-wise pairs with scale exchange."""
    bg = random.choice(BG_CHOICES)
    proto_a, proto_b = _make_prototypes(bg)
    protos = [proto_a, proto_b]

    # Top row: A@1 on the left, B@2 on the right.
    # Bottom row: A@2 on the left, B@1 on the right.
    # Output swaps each row pair, so the sizes move across the row.
    slots = [
        _slot_record(target_type=1, target_scale=BIG_SCALE, source_type=0, source_scale=SMALL_SCALE, protos=protos, bg=bg),
        _slot_record(target_type=0, target_scale=SMALL_SCALE, source_type=1, source_scale=BIG_SCALE, protos=protos, bg=bg),
        _slot_record(target_type=1, target_scale=SMALL_SCALE, source_type=0, source_scale=BIG_SCALE, protos=protos, bg=bg),
        _slot_record(target_type=0, target_scale=BIG_SCALE, source_type=1, source_scale=SMALL_SCALE, protos=protos, bg=bg),
    ]

    row_heights = [
        max(slots[0]["uh"], slots[1]["uh"]),
        max(slots[2]["uh"], slots[3]["uh"]),
    ]
    col_widths = [
        max(slots[0]["uw"], slots[2]["uw"]),
        max(slots[1]["uw"], slots[3]["uw"]),
    ]

    need_h = 2 * GRID_BORDER + row_heights[0] + row_heights[1] + ROW_GAP
    need_w = 2 * GRID_BORDER + col_widths[0] + col_widths[1] + COL_GAP
    n_min = max(GRID_SIZE_MIN, need_h, need_w)
    if n_min > GRID_SIZE_MAX:
        return None
    n = random.randint(n_min, GRID_SIZE_MAX)

    inp = np.full((n, n), bg, dtype=int)
    out = np.full((n, n), bg, dtype=int)

    top_margin = GRID_BORDER + random.randint(0, n - need_h)
    left_margin = GRID_BORDER + random.randint(0, n - need_w)
    row_bases = [top_margin, top_margin + row_heights[0] + ROW_GAP]
    col_bases = [left_margin, left_margin + col_widths[0] + COL_GAP]

    cell_map = {
        0: (0, 0),
        1: (0, 1),
        2: (1, 0),
        3: (1, 1),
    }

    for idx, slot in enumerate(slots):
        row_idx, col_idx = cell_map[idx]
        cell_r = row_bases[row_idx]
        cell_c = col_bases[col_idx]
        slack_r = row_heights[row_idx] - slot["uh"]
        slack_c = col_widths[col_idx] - slot["uw"]
        union_r = cell_r + random.randint(0, slack_r)
        union_c = cell_c + random.randint(0, slack_c)

        out_r, out_c, in_r, in_c = _place_from_union_origin(union_r, union_c, slot["dr"], slot["dc"])
        paste(out, slot["output_patch"], out_r, out_c, transparent=bg)
        paste(inp, slot["input_patch"], in_r, in_c, transparent=bg)

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
    raise RuntimeError("failed to generate a valid c7f57c3e D7 input")

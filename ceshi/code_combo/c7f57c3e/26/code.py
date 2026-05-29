# Combo variant of c7f57c3e (index 26)
# combo_spec: D3/2+D7/0+D8/2
# Name: 跨尺度缩放 + 按尺度排列位置 + 网格扩展
#
# Design analysis:
# spec:
#   preserved_structure:
#     - 仍然只有两种多色原型，且每个精灵都是某个原型的整数最近邻缩放副本。
#     - 颜色 1 仍然构成稳定锚框，因此 solver 仍可用锚框尺寸恢复 scale，再用降采样恢复逻辑原型。
#     - 最小尺度精灵仍必须同时给出两种原型，所以生成器固定保留一个 type0@1 和一个 type1@1。
#   fusion_rule_concretization:
#     - D7/0 若按字面理解成“源图案缩放到目标槽位原尺度”，会退化成原题的“每个位置仍保留自己的尺度”，
#       与 D3/D8 叠加后几乎不可见。
#     - 因此这里沿用单变体参考中的可观察实现：一对异尺度精灵交换时，原型和尺度作为一个 bundle 跨配对移动。
#       也就是 pair(A@s1, B@s2) 会产出 [B@s2, A@s1]，而不是 [B@s1, A@s2]。
#     - D3/2 不再把这些结果贴回原位，而是把“已交换 bundle 的四个结果”按 scale 从小到大排成一行。
#       同尺度并列时，用输入里的可见槽位顺序（先上后下、同行内先左后右）打破平局，不使用颜色 id。
#     - D8/2 进一步要求输出不再沿用输入方形画布，而是新建一个恰好容纳排序后单行精灵的矩形网格。
#   behavior_trace:
#     - 输入固定是 2x2 配对布局，每行恰好一对异原型精灵。
#     - 具体例子：上排 [A@1, B@3]，下排 [A@2, B@1]。
#     - 原题若逐位置同尺度换型，会得到 [B@1, A@3; B@2, A@1]，再怎么排位置都看不到跨尺度 bundle。
#     - 本融合版先做跨尺度配对交换，得到四个“待输出 bundle”：
#       top-left <- B@3, top-right <- A@1, bottom-left <- B@1, bottom-right <- A@2。
#     - 再按 scale 排成一行，输出次序变成 [A@1, B@1, A@2, B@3]；最后网格宽度随这四个 bbox 总宽度自动增长。
#     - 差异点有三层联动：谁变成谁（换原型）、变成多大（跨尺度）、最后落在哪（按尺度排序而非原位）。
#   complexity_calibration:
#     - 只使用四个精灵，避免 6 个以上对象导致输出行过宽、视觉杂乱。
#     - 尺度档位固定为 {1,2,3}，并固定两组尺度多重集为 [1,2] 与 [1,3]。
#       这样既保留两种最小原型，又让 D7 真正改变“交换后结果里每种原型对应哪些尺度”。
#     - 原型宽度限制为 3 列，确保 scale=3 的最宽输出行仍在可读范围内，同时足以让网格扩展肉眼可见。
#   hidden_assumptions_review:
#     - 原 solver 逐个组件独立就地回贴，依赖“输出位置就是当前 bbox 附近”；D3/2 后该前提失效。
#     - 原 solver 用最小面积组件恢复两种原型；这里继续保证恰有两个 scale=1 组件且类型不同，因此该前提保留。
#     - 新 solver 依赖“每种原型各出现两次，且按锚点行序可稳定配对”；生成器固定成两行配对布局来保证。
#     - 新 solver 依赖“每个组件都是真正的整数 NN 缩放”；生成器直接从原型缩放构造，避免近似缩放歧义。
#     - 新 solver 依赖“同尺度平局可由可见槽位顺序稳定打破”；生成器的 2x2 行列布局让该顺序肉眼可见。
#   legality:
#     - 输入画布中四个精灵放在 2x2 不重叠单元格里，单元格尺寸按该行/列最大输入 patch 预留，并带固定间距。
#     - 输出画布单独新建，按排序后的 patch 宽度总和直接计算尺寸，因此不会越界，也不会出现覆盖冲突。
#   observability:
#     - 解题者能直接看到：输入是两行配对，输出是单行重排；而且最大的精灵总被排到最右侧。
#     - 同时，若错误地忽略 D7，只做“同尺度换型后排序”，会把某些原型落在错误的尺度档位上；
#       这种差异在 2-3 个样例里是稳定可见的。
#   neatness_and_color_inference:
#     - 颜色变化仍然只来自“两种原型互换”，没有引入额外颜色编码。
#     - 输出中的四个精灵彼此分离、无遮挡、顶边对齐，便于同时观察形状类别、尺度和排序规则。

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


def _pick_anchor_bbox(arr: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    """Return the bbox of color-1 anchor pixels."""
    ys, xs = np.where(arr == 1)
    assert ys.size > 0, "every sprite must contain color-1 anchor pixels"
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
    assert h % scale == 0 and w % scale == 0, "scaled sprite must be divisible by scale"
    return arr[::scale, ::scale].copy()


def _recover_prototypes(sprites: list[dict[str, object]]) -> list[np.ndarray]:
    """Recover the two base prototypes from the smallest-area sprites."""
    areas = [
        int((int(s["bbox"][2]) - int(s["bbox"][0]) + 1) * (int(s["bbox"][3]) - int(s["bbox"][1]) + 1))
        for s in sprites
    ]
    min_area = min(areas)
    base = [s for s, area in zip(sprites, areas, strict=True) if area == min_area]

    uniq: dict[tuple[int, ...], np.ndarray] = {}
    for s in base:
        arr = np.asarray(s["arr"], dtype=int)
        key = tuple(arr.ravel().tolist())
        if key not in uniq:
            uniq[key] = arr.copy()
    assert len(uniq) == 2, "smallest sprites must provide exactly two prototype patterns"
    return [uniq[key] for key in list(uniq.keys())[:2]]


def _classify_sprites(
    sprites: list[dict[str, object]], protos: list[np.ndarray], bg: int
) -> list[dict[str, object]]:
    """Attach prototype id, scale, and visible slot order metadata to each sprite."""
    anc0 = _pick_anchor_bbox(protos[0], bg)
    anc1 = _pick_anchor_bbox(protos[1], bg)
    base_anchor_h0 = anc0[2] - anc0[0] + 1
    base_anchor_w0 = anc0[3] - anc0[1] + 1
    base_anchor_h1 = anc1[2] - anc1[0] + 1
    base_anchor_w1 = anc1[3] - anc1[1] + 1
    assert (base_anchor_h0, base_anchor_w0) == (base_anchor_h1, base_anchor_w1)

    for s in sprites:
        arr = np.asarray(s["arr"], dtype=int)
        r0, c0, _r1, _c1 = s["bbox"]
        ar0, ac0, ar1, ac1 = _pick_anchor_bbox(arr, bg)
        inst_anchor_h = ar1 - ar0 + 1
        inst_anchor_w = ac1 - ac0 + 1
        assert inst_anchor_h % base_anchor_h0 == 0 and inst_anchor_w % base_anchor_w0 == 0
        scale_h = inst_anchor_h // base_anchor_h0
        scale_w = inst_anchor_w // base_anchor_w0
        assert scale_h == scale_w and scale_h >= 1
        scale = int(scale_h)

        logical = _descale_nn(arr, scale)
        proto_idx = None
        for idx, proto in enumerate(protos):
            if np.array_equal(logical, proto):
                proto_idx = idx
                break
        assert proto_idx is not None, "sprite must descale to one of the base prototypes"

        s["type"] = int(proto_idx)
        s["scale"] = int(scale)
        s["anchor_global"] = (int(r0) + int(ar0), int(c0) + int(ac0))

    ordered_slots = sorted(
        sprites,
        key=lambda s: (
            int(s["anchor_global"][0]),
            int(s["anchor_global"][1]),
            int(s["bbox"][0]),
            int(s["bbox"][1]),
        ),
    )
    for slot_rank, s in enumerate(ordered_slots):
        s["slot_rank"] = int(slot_rank)
    return sprites


def _transformed_records(
    sprites: list[dict[str, object]],
) -> list[dict[str, int]]:
    """Apply cross-scale pair swapping and keep the target slot rank for later ordering."""
    groups = {0: [], 1: []}
    for s in sprites:
        groups[int(s["type"])].append(s)
    assert len(groups[0]) == len(groups[1]) == 2, "combo layout requires two sprites of each prototype"

    for idx in (0, 1):
        groups[idx].sort(key=lambda s: (int(s["anchor_global"][0]), int(s["anchor_global"][1])))

    transformed: list[dict[str, int]] = []
    for sa, sb in zip(groups[0], groups[1], strict=True):
        transformed.append(
            {
                "slot_rank": int(sa["slot_rank"]),
                "type": int(sb["type"]),
                "scale": int(sb["scale"]),
            }
        )
        transformed.append(
            {
                "slot_rank": int(sb["slot_rank"]),
                "type": int(sa["type"]),
                "scale": int(sa["scale"]),
            }
        )
    return transformed


def _render_sorted_row(
    protos: list[np.ndarray], bg: int, records: list[dict[str, int]], border: int, gap: int
) -> np.ndarray:
    """Render the transformed sprites as one left-to-right row sorted by scale."""
    ordered = sorted(records, key=lambda rec: (int(rec["scale"]), int(rec["slot_rank"])))
    patches = [_scale_nn(protos[int(rec["type"])], int(rec["scale"])) for rec in ordered]
    out_h = 2 * int(border) + max(p.shape[0] for p in patches)
    out_w = 2 * int(border) + sum(int(p.shape[1]) for p in patches) + int(gap) * (len(patches) - 1)
    out = np.full((int(out_h), int(out_w)), int(bg), dtype=int)

    cur_c = int(border)
    for patch in patches:
        paste(out, patch, int(border), int(cur_c), transparent=bg)
        cur_c += int(patch.shape[1]) + int(gap)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair opposite prototypes across rows, swap prototype+scale bundles, then lay the results out in scale order on an expanded grid.

    Input structure:
    - The background is uniform. Exactly four connected sprites appear, all built from the same two multicolor prototypes.
    - Each prototype appears twice at integer scales, and the four sprites occupy a visible 2x2 pairing layout with one sprite of each prototype in each row.

    Rule:
    - Recover the two base prototypes from the smallest sprites, then infer every sprite's prototype type and integer scale from its color-1 anchor box.
    - Within each prototype, sort sprites from top to bottom and pair the k-th sprite of one prototype with the k-th sprite of the other; each pair exchanges prototype and scale together.
    - Take the four exchanged results, sort them globally by scale from small to large, break ties by the original visible slot order, and place them left to right on a newly sized output grid.

    Key insight: Scale no longer belongs to the destination slot, and position no longer belongs to the original component; first the pair determines which prototype-scale bundle exists, then the bundle's scale determines where it lands in the expanded row.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=8)
    assert len(comps) == 4, "combo variant expects exactly four sprite components"

    sprites: list[dict[str, object]] = []
    for idx, pts in enumerate(comps):
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        sprites.append({"index": int(idx), "pts": pts, "bbox": (r0, c0, r1, c1), "arr": arr})

    protos = _recover_prototypes(sprites)
    _classify_sprites(sprites, protos, bg)
    transformed = _transformed_records(sprites)
    return _render_sorted_row(protos, bg, transformed, border=OUT_BORDER, gap=OUT_GAP)


# --------------- generator (combo) ---------------

# --- CFG: canvas ---
GRID_SIZE_MIN = 18
GRID_SIZE_MAX = 28
INPUT_BORDER = 1
INPUT_ROW_GAP = 2
INPUT_COL_GAP = 2
OUT_BORDER = 1
OUT_GAP = 1

# --- CFG: prototypes ---
PROTO_SHAPES = [(3, 3), (4, 3)]
BG_CHOICES = [0, 2, 3, 4, 5, 6, 7, 8, 9]
COLOR_SHARE_PROB = 0.65

# --- CFG: row-mask weights ---
ANCHOR_ROW_WEIGHTS = [("center", 0.45), ("outer", 0.35), ("full", 0.20)]
VARIABLE_ROW_WEIGHTS = [("center", 0.68), ("full", 0.32)]
OUTER_FIXUP_PROB = 0.7

# --- CFG: generation retries ---
MAX_RETRIES = 300

_ROW_MASKS = {
    "center": {3: [0, 1, 0]},
    "outer": {3: [1, 0, 1]},
    "full": {3: [1, 1, 1]},
}


def _row_mask(kind: str, w: int) -> list[int]:
    return list(_ROW_MASKS[kind][w])


def _make_labels() -> np.ndarray:
    """Build a small connected label grid with roles 1, 2, 3 and visible bg holes."""
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
        kinds, weights = zip(*weights_by_role[role])
        masks.append(_row_mask(random.choices(kinds, weights=weights)[0], w))

    if not any(mask[0] or mask[-1] for mask in masks):
        idx = random.randrange(h)
        if roles[idx] == 1 and random.random() < OUTER_FIXUP_PROB:
            masks[idx] = _row_mask("outer", w)
        else:
            masks[idx] = _row_mask("full", w)

    if all(all(mask) for mask in masks):
        masks[random.randrange(h)] = _row_mask("center", w)

    labels = np.zeros((h, w), dtype=int)
    for r, (role, mask) in enumerate(zip(roles, masks, strict=True)):
        for c, bit in enumerate(mask):
            if bit:
                labels[r, c] = role

    if len(components_mask(labels != 0, connectivity=8)) != 1:
        return _make_labels()
    return labels


def _make_prototypes(bg: int) -> tuple[np.ndarray, np.ndarray]:
    """Create proto_A and proto_B from one structural label grid."""
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


def _build_layout_assignments() -> list[tuple[int, int]]:
    """Return row-major input assignments for the 2x2 pairing layout."""
    rows = [[(0, 1), (1, 3)], [(0, 2), (1, 1)]]
    if random.random() < 0.5:
        rows = [list(reversed(row)) for row in rows]
    if random.random() < 0.5:
        rows.reverse()
    return [rows[0][0], rows[0][1], rows[1][0], rows[1][1]]


def _render_input_scene(n: int, bg: int, patches: list[np.ndarray]) -> np.ndarray:
    """Place four input patches inside a 2x2 non-overlapping layout on an n x n grid."""
    row_heights = [
        max(int(patches[0].shape[0]), int(patches[1].shape[0])),
        max(int(patches[2].shape[0]), int(patches[3].shape[0])),
    ]
    col_widths = [
        max(int(patches[0].shape[1]), int(patches[2].shape[1])),
        max(int(patches[1].shape[1]), int(patches[3].shape[1])),
    ]
    need_h = 2 * INPUT_BORDER + row_heights[0] + row_heights[1] + INPUT_ROW_GAP
    need_w = 2 * INPUT_BORDER + col_widths[0] + col_widths[1] + INPUT_COL_GAP
    assert need_h <= n and need_w <= n

    extra_h = int(n - need_h)
    extra_w = int(n - need_w)
    top0 = INPUT_BORDER + random.randint(0, extra_h)
    left0 = INPUT_BORDER + random.randint(0, extra_w)
    row_bases = [top0, top0 + row_heights[0] + INPUT_ROW_GAP]
    col_bases = [left0, left0 + col_widths[0] + INPUT_COL_GAP]

    grid = np.full((n, n), int(bg), dtype=int)
    for idx, patch in enumerate(patches):
        rr = idx // 2
        cc = idx % 2
        cell_r = row_bases[rr]
        cell_c = col_bases[cc]
        slack_r = row_heights[rr] - int(patch.shape[0])
        slack_c = col_widths[cc] - int(patch.shape[1])
        r0 = cell_r + random.randint(0, slack_r)
        c0 = cell_c + random.randint(0, slack_c)
        paste(grid, patch, int(r0), int(c0), transparent=bg)
    return grid


def _expected_output(
    protos: list[np.ndarray], bg: int, assignments: list[tuple[int, int]]
) -> np.ndarray:
    """Ground-truth renderer for the fused rule, independent of the solver."""
    swapped = [assignments[1], assignments[0], assignments[3], assignments[2]]
    records = [
        {"slot_rank": 0, "type": int(swapped[0][0]), "scale": int(swapped[0][1])},
        {"slot_rank": 1, "type": int(swapped[1][0]), "scale": int(swapped[1][1])},
        {"slot_rank": 2, "type": int(swapped[2][0]), "scale": int(swapped[2][1])},
        {"slot_rank": 3, "type": int(swapped[3][0]), "scale": int(swapped[3][1])},
    ]
    return _render_sorted_row(protos, bg, records, border=OUT_BORDER, gap=OUT_GAP)


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_CHOICES)
    proto_a, proto_b = _make_prototypes(bg)
    protos = [proto_a, proto_b]
    assignments = _build_layout_assignments()
    patches = [_scale_nn(protos[typ], scale) for typ, scale in assignments]
    out = _expected_output(protos, bg, assignments)

    row_heights = [
        max(int(patches[0].shape[0]), int(patches[1].shape[0])),
        max(int(patches[2].shape[0]), int(patches[3].shape[0])),
    ]
    col_widths = [
        max(int(patches[0].shape[1]), int(patches[2].shape[1])),
        max(int(patches[1].shape[1]), int(patches[3].shape[1])),
    ]
    need_h = 2 * INPUT_BORDER + row_heights[0] + row_heights[1] + INPUT_ROW_GAP
    need_w = 2 * INPUT_BORDER + col_widths[0] + col_widths[1] + INPUT_COL_GAP

    lower = max(GRID_SIZE_MIN, need_h, need_w)
    upper = min(GRID_SIZE_MAX, int(out.shape[1]) - 1)
    if lower > upper:
        return None

    n = random.randint(lower, upper)
    inp = _render_input_scene(n, bg, patches)

    try:
        solved = main(inp)
    except Exception:
        return None
    if solved.shape[1] <= inp.shape[1]:
        return None
    if np.array_equal(inp, solved):
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid c7f57c3e combo input")

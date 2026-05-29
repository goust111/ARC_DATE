# Combo variant of c7f57c3e (index 9)
# combo_spec: D2/0+D7/0
# Name: 交换时缩放到目标尺度 + 仅按距离配对
#
# Design analysis
# spec:
#   concrete_behavior_trace:
#     original: |
#       原题可视为“识别两种精灵原型，然后每个位置独立切到另一种原型，尺度仍取当前位置自己的尺度”。
#       即使口头上说“跨组匹配”，在原始生成分布里同组实例的颜色模式完全一致，
#       所以匹配顺序不会影响结果；每个槽位只会看到“同尺度换花纹”。
#     combo: |
#       这个融合版把“配对”和“缩放”真正绑在一起：
#       先按跨组最近距离做一一配对，不再要求同尺度；
#       再把“配对源精灵的完整颜色图案”缩放到目标槽位自己的尺度后重绘到目标位置。
#       为了让配对确实可见，生成器让同一结构组内的两个实例带不同强调色；
#       因此 A 小精灵若最近的是 B 大精灵，输出到 A 槽位的就是“B 结构 + B 大精灵的强调色”，
#       但尺寸仍保持 A 槽位的小尺度。
#   small_example:
#     trace: |
#       设四个槽位是 TL:A@1(a), TR:B@2(d), BL:B@1(c), BR:A@2(b)。
#       距离最近配对被布局成同一行：TL 配 TR，BR 配 BL，所以是异尺度配对。
#       输出因此变成 TL<-B(d)@1，TR<-A(a)@2，BL<-A(b)@1，BR<-B(c)@2。
#       若仍按原题“同尺度对应”，则会得到 TL<-B(c)@1，TR<-A(b)@2，BL<-A(a)@1，BR<-B(d)@2。
#       两者差异正好落在“谁和谁配对”，而 D7 的目标尺度规则保证尺寸仍由槽位自己决定。
#   complexity_check:
#     note: |
#       只保留 4 个精灵、两档尺度（1 和 2）、每组各 2 个实例，避免距离匹配过于混乱；
#       同时固定每个样例都发生两次异尺度最近配对，并让同组强调色不同，
#       这样 2-3 个样例就能看出“最近的异组邻居决定被换入的颜色模式”。
#   hidden_assumptions_review:
#     solver_dependencies:
#       - "最小面积精灵恰好提供两种 scale=1 结构原型；生成器固定四槽布局并保证只有 A@1 与 B@1 最小。"
#       - "所有实例都是整数最近邻缩放；solver 用 anchor bbox 尺寸恢复 scale，再降采样恢复逻辑图案。"
#       - "anchor 色始终是 1，且 role-2 使用固定颜色 2；solver 以此把结构归一化为 {bg,1,2,accent}。"
#       - "两组数量相等，且最近距离匹配唯一；生成器只造两对并通过行布局和验证步骤排除歧义。"
#       - "交换后可能因 A/B anchor 偏移不同而相对原 bbox 平移；因此每个槽位按 input/output 联合包围盒预留空间。"
#     multi_change_conflicts:
#       - "若取消同尺度约束但同组内部完全同色，则 D2 会退化成原题；这里用组内不同强调色消除退化。"
#       - "若让源尺度一起移动，就会落回 D7 单变体；这里严格采用'源图案重采样到目标尺度'。"
#   legality_observability_cleanliness:
#     placement: |
#       生成器使用 2x2 槽位布局，按每槽 input/output 联合包围盒计算行高列宽，再加固定边框和间距；
#       因而输入输出都不会越界或重叠。
#     observability: |
#       配对只依赖可见空间距离（anchor 位置），不用颜色 id 排序等不可见信息。
#       每张图都强制最近配对为异尺度、且与“同尺度配对基线”输出不同。
#     visual_cleanliness: |
#       结构仍然是原题式的小型连通精灵，颜色种类控制在 bg + 1 + 2 + 4 accents 以内，
#       没有叠写标记或遮挡，四个对象分布清晰。

import itertools
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


CORE_COLOR = 2


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
    """Recover the logical sprite from an exact NN-scaled copy."""
    scale = int(scale)
    if scale <= 1:
        return arr.copy()
    h, w = arr.shape
    assert h % scale == 0 and w % scale == 0, "sprite shape must be divisible by scale"
    return arr[::scale, ::scale].copy()


def _normalize_structure(arr: np.ndarray, bg: int) -> np.ndarray:
    """Normalize a colored logical sprite to {bg, 1, 2, 3} for structure matching."""
    out = np.full(arr.shape, bg, dtype=int)
    out[arr == 1] = 1
    out[arr == CORE_COLOR] = CORE_COLOR
    accent = (arr != bg) & (arr != 1) & (arr != CORE_COLOR)
    out[accent] = 3
    return out


def _anchor_center2(sprite: dict) -> tuple[int, int]:
    """Return doubled anchor-center coordinates to compare distances without floats."""
    gr0 = sprite["bbox"][0] + sprite["anchor_bbox"][0]
    gc0 = sprite["bbox"][1] + sprite["anchor_bbox"][1]
    gr1 = sprite["bbox"][0] + sprite["anchor_bbox"][2]
    gc1 = sprite["bbox"][1] + sprite["anchor_bbox"][3]
    return int(gr0 + gr1), int(gc0 + gc1)


def _extract_sprites(g: np.ndarray) -> tuple[int, list[dict], list[np.ndarray]]:
    """Parse sprites, infer structure type/scale, and return normalized base structures."""
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
                "anchor_tl": (int(r0 + anc[0]), int(c0 + anc[1])),
            }
        )

    areas = [int((s["bbox"][2] - s["bbox"][0] + 1) * (s["bbox"][3] - s["bbox"][1] + 1)) for s in sprites]
    min_area = min(areas)
    base = [s for s, a in zip(sprites, areas, strict=False) if a == min_area]

    base_structs: list[np.ndarray] = []
    base_anchor_shape: tuple[int, int] | None = None
    for s in base:
        norm = _normalize_structure(s["arr"], bg)
        if not any(np.array_equal(norm, seen) for seen in base_structs):
            base_structs.append(norm)
            anc = _pick_anchor_bbox(s["arr"], bg)
            anc_shape = (anc[2] - anc[0] + 1, anc[3] - anc[1] + 1)
            if base_anchor_shape is None:
                base_anchor_shape = anc_shape
            else:
                assert anc_shape == base_anchor_shape, "base structures must share anchor size"
    assert len(base_structs) == 2, "smallest sprites must reveal exactly two normalized structures"
    assert base_anchor_shape is not None
    base_anchor_h, base_anchor_w = base_anchor_shape

    for s in sprites:
        ar0, ac0, ar1, ac1 = s["anchor_bbox"]
        inst_anchor_h = ar1 - ar0 + 1
        inst_anchor_w = ac1 - ac0 + 1
        assert inst_anchor_h % base_anchor_h == 0 and inst_anchor_w % base_anchor_w == 0
        scale_h = inst_anchor_h // base_anchor_h
        scale_w = inst_anchor_w // base_anchor_w
        assert scale_h == scale_w and scale_h >= 1, "scale must be a positive integer"
        scale = int(scale_h)

        logical = _descale_nn(s["arr"], scale)
        norm = _normalize_structure(logical, bg)
        match_idx = None
        for idx, proto in enumerate(base_structs):
            if np.array_equal(norm, proto):
                match_idx = idx
                break
        assert match_idx is not None, "every sprite must match one of the two normalized structures"
        s["type"] = int(match_idx)
        s["scale"] = scale
        s["logical"] = logical
        s["center2"] = _anchor_center2(s)

    return bg, sprites, base_structs


def _match_pairs_by_distance(group_a: list[dict], group_b: list[dict]) -> list[tuple[dict, dict]]:
    """Find the unique minimum-total-distance matching between the two groups."""
    assert len(group_a) == len(group_b) >= 1
    best_pairs = None
    best_cost = None
    best_perm = None
    for perm in itertools.permutations(range(len(group_b))):
        pairs = [(group_a[i], group_b[j]) for i, j in enumerate(perm)]
        cost = sum(
            (sa["center2"][0] - sb["center2"][0]) ** 2 + (sa["center2"][1] - sb["center2"][1]) ** 2
            for sa, sb in pairs
        )
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_pairs = pairs
            best_perm = perm
        elif cost == best_cost:
            assert perm == best_perm, "distance matching must be unique"
    assert best_pairs is not None
    return best_pairs


def _same_scale_pairs(group_a: list[dict], group_b: list[dict]) -> list[tuple[dict, dict]]:
    """Reference pairing used only to detect generator degeneracy."""
    a_by_scale = {s["scale"]: s for s in group_a}
    b_by_scale = {s["scale"]: s for s in group_b}
    assert set(a_by_scale) == set(b_by_scale), "same-scale reference requires one counterpart per scale"
    return [(a_by_scale[scale], b_by_scale[scale]) for scale in sorted(a_by_scale)]


def _render_pairs(g: np.ndarray, bg: int, pairs: list[tuple[dict, dict]]) -> np.ndarray:
    """Render swapped output for a chosen pairing relation."""
    out = np.full_like(g, bg)
    for sa, sb in pairs:
        patch_for_a = _scale_nn(sb["logical"], sa["scale"])
        patch_for_b = _scale_nn(sa["logical"], sb["scale"])

        anc_patch_a = _pick_anchor_bbox(patch_for_a, bg)
        anc_patch_b = _pick_anchor_bbox(patch_for_b, bg)

        top_a = sa["anchor_tl"][0] - anc_patch_a[0]
        left_a = sa["anchor_tl"][1] - anc_patch_a[1]
        top_b = sb["anchor_tl"][0] - anc_patch_b[0]
        left_b = sb["anchor_tl"][1] - anc_patch_b[1]

        paste(out, patch_for_a, top_a, left_a, transparent=bg)
        paste(out, patch_for_b, top_b, left_b, transparent=bg)
    return out


def _same_scale_baseline(grid: np.ndarray) -> np.ndarray:
    """Baseline output for the old same-scale pairing, used for observability checks."""
    g = np.asarray(grid, dtype=int)
    bg, sprites, _ = _extract_sprites(g)
    groups = {0: [], 1: []}
    for s in sprites:
        groups[s["type"]].append(s)
    return _render_pairs(g, bg, _same_scale_pairs(groups[0], groups[1]))


def _aligned_union_bbox(
    output_patch: np.ndarray, input_patch: np.ndarray, bg: int
) -> tuple[int, int, int, int]:
    """Union box for anchor-aligned output/input placement at one slot."""
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
    """Convert union origin plus relative offset into output/input top-lefts."""
    out_r = union_r - min(0, dr)
    out_c = union_c - min(0, dc)
    in_r = out_r + dr
    in_c = out_c + dc
    return int(out_r), int(out_c), int(in_r), int(in_c)


def _colorize(labels: np.ndarray, bg: int, accent: int) -> np.ndarray:
    """Instantiate one logical sprite from structural labels."""
    arr = np.full(labels.shape, bg, dtype=int)
    arr[labels == 1] = 1
    arr[labels == 2] = CORE_COLOR
    arr[labels == 3] = accent
    return arr


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap each sprite with its nearest opposite-group partner, then rescale the source pattern to the target slot.

    Input structure:
    - The background is the most common color. There are exactly two sprite structures, each appearing twice.
    - Every sprite is an integer nearest-neighbor scaling of one of those structures and includes a color-1 anchor region.
    - Sprites from the same structure may use different accent colors even when their geometry matches.

    Rule:
    - Recover the two base structures from the smallest sprites, infer each sprite's structure type, scale, and logical color pattern.
    - Split sprites into the two structure groups and find the minimum-distance one-to-one cross-group matching using anchor positions only; scale is ignored during pairing.
    - For each matched pair, redraw each source sprite's full logical color pattern at the partner's location, but resize it to the partner slot's scale and align by the color-1 anchor.

    Key insight: The pairing source decides which exact color pattern moves, while the target slot decides the final size; nearest-distance matching and target-scale repainting must both be inferred.
    """
    g = np.asarray(grid, dtype=int)
    bg, sprites, _ = _extract_sprites(g)

    groups = {0: [], 1: []}
    for s in sprites:
        groups[s["type"]].append(s)
    assert len(groups[0]) == len(groups[1]) >= 1, "the two structure groups must have equal size"

    pairs = _match_pairs_by_distance(groups[0], groups[1])
    return _render_pairs(g, bg, pairs)


# --------------- generator (o5x) ---------------

# --- CFG: canvas ---
GRID_SIZE_MIN = 18
GRID_SIZE_MAX = 28
GRID_BORDER = 1
ROW_GAP = 3
COL_GAP = 3

# --- CFG: prototypes ---
PROTO_SHAPES = [(4, 3), (3, 4), (4, 4)]
BG_CHOICES = [0, 3, 4, 5, 6, 7, 8, 9]
ACCENT_POOL = [3, 4, 5, 6, 7, 8, 9]

# --- CFG: row-mask weights ---
ANCHOR_ROW_WEIGHTS = [("center", 0.45), ("outer", 0.35), ("full", 0.20)]
CORE_ROW_WEIGHTS = [("center", 0.55), ("full", 0.45)]
ACCENT_ROW_WEIGHTS = [("center", 0.68), ("full", 0.32)]
OUTER_FIXUP_PROB = 0.7

# --- CFG: scales ---
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
    """Build a connected label grid with roles 1(anchor), 2(core), 3(accent)."""
    h, w = random.choice(PROTO_SHAPES)

    counts = {1: 1, 2: 1, 3: 1}
    if h > 3:
        counts[random.choice([1, 2, 3])] += h - 3

    roles = []
    for role in random.sample([1, 2, 3], 3):
        roles.extend([role] * counts[role])

    weights_by_role = {
        1: ANCHOR_ROW_WEIGHTS,
        2: CORE_ROW_WEIGHTS,
        3: ACCENT_ROW_WEIGHTS,
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
    if not np.any(labels == 2) or not np.any(labels == 3):
        return _make_labels()
    if np.array_equal(labels, np.flipud(labels)):
        return _make_labels()
    return labels


def _make_structures() -> tuple[np.ndarray, np.ndarray]:
    """Create the two structural prototypes as label grids."""
    labels = _make_labels()
    return labels.copy(), np.flipud(labels).copy()


def _slot_record(
    target_type: int,
    target_scale: int,
    target_accent: int,
    source_type: int,
    source_accent: int,
    structures: list[np.ndarray],
    bg: int,
) -> dict:
    """Describe one slot together with its input/output patches and reserved box."""
    input_logical = _colorize(structures[target_type], bg, target_accent)
    output_logical = _colorize(structures[source_type], bg, source_accent)
    input_patch = _scale_nn(input_logical, target_scale)
    output_patch = _scale_nn(output_logical, target_scale)
    dr, dc, uh, uw = _aligned_union_bbox(output_patch, input_patch, bg)
    return {
        "input_patch": input_patch,
        "output_patch": output_patch,
        "dr": dr,
        "dc": dc,
        "uh": uh,
        "uw": uw,
    }


def _distance_pairs_are_cross_scale(inp: np.ndarray) -> bool:
    """Check that distance-only matching really ignores scale on this sample."""
    _bg, sprites, _ = _extract_sprites(inp)
    groups = {0: [], 1: []}
    for s in sprites:
        groups[s["type"]].append(s)
    pairs = _match_pairs_by_distance(groups[0], groups[1])
    return all(sa["scale"] != sb["scale"] for sa, sb in pairs)


def _gen_attempt() -> np.ndarray | None:
    """Construct one valid sample with visible nearest-distance cross-scale pairing."""
    bg = random.choice(BG_CHOICES)
    if bg == CORE_COLOR:
        return None

    accents = random.sample([c for c in ACCENT_POOL if c != bg], 4)
    a_small, a_big, b_small, b_big = accents

    struct_a, struct_b = _make_structures()
    structures = [struct_a, struct_b]

    # Row-wise nearest pairs are intentionally cross-scale:
    # top row: A@1 is closest to B@2
    # bottom row: B@1 is closest to A@2
    slots = [
        _slot_record(
            target_type=0,
            target_scale=SMALL_SCALE,
            target_accent=a_small,
            source_type=1,
            source_accent=b_big,
            structures=structures,
            bg=bg,
        ),
        _slot_record(
            target_type=1,
            target_scale=BIG_SCALE,
            target_accent=b_big,
            source_type=0,
            source_accent=a_small,
            structures=structures,
            bg=bg,
        ),
        _slot_record(
            target_type=1,
            target_scale=SMALL_SCALE,
            target_accent=b_small,
            source_type=0,
            source_accent=a_big,
            structures=structures,
            bg=bg,
        ),
        _slot_record(
            target_type=0,
            target_scale=BIG_SCALE,
            target_accent=a_big,
            source_type=1,
            source_accent=b_small,
            structures=structures,
            bg=bg,
        ),
    ]

    row_heights = [max(slots[0]["uh"], slots[1]["uh"]), max(slots[2]["uh"], slots[3]["uh"])]
    col_widths = [max(slots[0]["uw"], slots[2]["uw"]), max(slots[1]["uw"], slots[3]["uw"])]

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
        combo = main(inp)
        baseline = _same_scale_baseline(inp)
        if not _distance_pairs_are_cross_scale(inp):
            return None
        if np.array_equal(combo, baseline):
            return None
        if np.array_equal(combo, out):
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

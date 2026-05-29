# Variant of c7f57c3e
# Dimension: D9 — 多步处理顺序 (multi_step)
# Change: 步骤：识别精灵 → 按颜色模式分组 → 同尺度跨组配对 → 交换颜色模式... → 条件性交换
#
# Design analysis
# a) 具体化与行为追踪
#    输入仍然是同一类多色精灵：两种原型，共享 color-1 锚点区域，其他颜色不同，实例以整数倍缩放后分散放置。
#    新变体把“是否交换”做成显式的前置判断：先找出最小 bbox 面积的那一批精灵，它们既提供两种原型，也定义阈值。
#    只有 bbox 面积严格大于该最小面积的精灵会交换到另一原型；最小尺寸精灵保持不变。
#    例子：若最小精灵 A、B 都是 4x4，另外还有 8x8 的 A 和 12x12 的 B，则原题会四个都交换；
#    本变体中 4x4 的 A/B 不变，8x8 的 A 变成 B，12x12 的 B 变成 A。
#    差异体现在流程顺序上：原题“识别后直接全换”，这里必须先判断参与集合，再只对参与者做匹配和替换。
# b) 复杂度校准
#    条件只用“相对尺寸”这一种可见因素，档位很少：最小尺寸 vs 放大尺寸。这样不会引入难以观察的多分支。
#    同时生成器固定至少 2 个不参与者和 2 个参与者，避免只有 1 个对象变化导致视觉上像偶然。
# c) 隐式假设审查
#    原 main() 中 `repl_idx = 1 - best_idx` 默认每个精灵都要替换；在变体里这不成立，因此必须把“清空当前精灵”和
#    “粘贴替代原型”放到条件分支里，只对参与者执行。否则会把本应保留的小精灵错误改写。
#    原代码用最小面积精灵恢复两个原型，这个前提仍成立，但生成器必须保证最小面积层里确实有两种不同原型。
#    原代码的 scale 推断依赖锚点 bbox 在两个原型中同尺寸；生成器继续沿用共享 label + flipUD 的构造来保证这点。
# d) 放置合法性
#    仍使用 output-first：先决定输出里每个精灵最终该是什么，再反推输入。
#    对于参与交换的精灵，输入放相反原型；对于不参与的精灵，输入直接放输出同型原型。
#    放置时继续按锚点对齐后的并集 bbox 预留空间，保证输入/输出都不重叠、不越界，并且 solver 的对齐逻辑有效。
# e) 视觉可观察性
#    规则不依赖颜色 id、随机排序或隐藏元数据，只依赖人眼可见的整体尺寸。
#    在 2-4 个样例里，人类能看到一致规律：小精灵始终不动，大精灵始终切换颜色模式。
# f) 可视化整洁度
#    不额外添加标记，仍然只放精灵本体；通过维持较大的间距和最多 4 个实例，避免画面拥挤。
# g) 颜色变化可推断性
#    颜色变化仍然只是在两种原型之间切换，结构锚点保持不变，变化范围受原型内部形状约束，人类可直接对比看出。
# h) 新的 main() 隐式假设与 generator 保证
#    `main()` 假设：至少两个最小面积精灵且它们给出恰好两种原型；所有更大精灵都是这两种原型的整数倍缩放；
#    锚点颜色 1 总存在；对象间彼此分离。`generate_input()` 会固定放入 scale-1 的 A/B 各一个，再放入两个 scale>=2 的实例，
#    并通过 oracle `np.array_equal(main(inp), out)` 校验这些假设在样本上成立。

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


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _infer_scale(arr: np.ndarray, base_anchor_h: int, base_anchor_w: int, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    anc = _pick_anchor_bbox(arr, bg)
    ar0, ac0, ar1, ac1 = anc
    inst_anc_h = ar1 - ar0 + 1
    inst_anc_w = ac1 - ac0 + 1
    scale_h = max(1, inst_anc_h // max(1, base_anchor_h))
    scale_w = max(1, inst_anc_w // max(1, base_anchor_w))
    return int(max(scale_h, scale_w)), anc


def _match_prototype(
    g: np.ndarray,
    bbox: tuple[int, int, int, int],
    protos: list[np.ndarray],
    scale: int,
    anc: tuple[int, int, int, int],
    bg: int,
) -> int:
    h, w = g.shape
    r0, c0, _r1, _c1 = bbox
    ar0, ac0, _ar1, _ac1 = anc
    best_idx = 0
    best_score = -1
    for idx, p in enumerate(protos):
        sp = _scale_nn(p, scale)
        pr0, pc0, _pr1, _pc1 = _pick_anchor_bbox(p, bg)
        pr0 *= scale
        pc0 *= scale
        top = (r0 + ar0) - pr0
        left = (c0 + ac0) - pc0

        ph, pw = sp.shape
        rr0 = max(0, top)
        cc0 = max(0, left)
        rr1 = min(h, top + ph)
        cc1 = min(w, left + pw)
        if rr0 >= rr1 or cc0 >= cc1:
            continue
        sub_g = g[rr0:rr1, cc0:cc1]
        sub_p = sp[(rr0 - top) : (rr1 - top), (cc0 - left) : (cc1 - left)]
        m = sub_p != bg
        score = int(np.sum(sub_g[m] == sub_p[m])) if np.any(m) else 0
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap only the enlarged sprites to the opposite prototype while preserving scale and position.

    Input structure:
    - bg is the most common color. Two distinct prototype patterns appear as separate sprite components at multiple sizes.
    - The smallest-bbox sprites show both prototype patterns directly; larger sprites are integer-scaled copies of one of them.

    Rule:
    - Recover the two prototypes from the smallest bbox-area sprites and use that minimum area as the participation threshold.
    - For each sprite, infer which prototype it matches and its integer scale from the color-1 anchor bbox.
    - Leave minimum-area sprites unchanged, but replace every larger sprite with the other prototype at the same scale, aligned by the anchor region.

    Key insight: The smallest sprites are both examples and exceptions: they reveal the two base patterns and define the size cutoff, so only enlarged copies toggle while the prototype-sized copies stay fixed.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=8)
    assert len(comps) >= 2, "grid must contain at least 2 sprite components"

    sprites = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        sprites.append({"pts": pts, "bbox": (r0, c0, r1, c1), "arr": arr})

    areas = [_bbox_area(s["bbox"]) for s in sprites]
    min_area = min(areas)
    base = [s for s, a in zip(sprites, areas, strict=False) if a == min_area]
    uniq: dict[tuple[int, ...], np.ndarray] = {}
    for s in base:
        key = tuple(s["arr"].ravel().tolist())
        if key not in uniq:
            uniq[key] = s["arr"]
    assert len(uniq) == 2, "smallest-area sprites must yield exactly 2 distinct prototype patterns"

    proto_keys = list(uniq.keys())[:2]
    protos = [uniq[proto_keys[0]], uniq[proto_keys[1]]]

    anc0 = _pick_anchor_bbox(protos[0], bg)
    anc1 = _pick_anchor_bbox(protos[1], bg)
    base_anc_h0 = anc0[2] - anc0[0] + 1
    base_anc_w0 = anc0[3] - anc0[1] + 1
    base_anc_h1 = anc1[2] - anc1[0] + 1
    base_anc_w1 = anc1[3] - anc1[1] + 1
    assert (base_anc_h0, base_anc_w0) == (base_anc_h1, base_anc_w1), "prototype anchors must share a size"

    out = g.copy()

    for s, area in zip(sprites, areas, strict=False):
        if area == min_area:
            continue

        r0, c0, _r1, _c1 = s["bbox"]
        arr = s["arr"]
        scale, anc = _infer_scale(arr, base_anc_h0, base_anc_w0, bg)
        best_idx = _match_prototype(g, s["bbox"], protos, scale, anc, bg)
        repl_idx = 1 - best_idx

        for rr, cc in s["pts"]:
            out[rr, cc] = bg

        p = protos[repl_idx]
        sp = _scale_nn(p, scale)
        pr0, pc0, _pr1, _pc1 = _pick_anchor_bbox(p, bg)
        pr0 *= scale
        pc0 *= scale
        ar0, ac0, _ar1, _ac1 = anc
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
COLOR_SHARE_PROB = 0.65

# --- CFG: row-mask weights (anchor role vs variable role) ---
ANCHOR_ROW_WEIGHTS = [("center", 0.45), ("outer", 0.35), ("full", 0.20)]
VARIABLE_ROW_WEIGHTS = [("center", 0.68), ("full", 0.32)]
OUTER_FIXUP_PROB = 0.7

# --- CFG: sprite instances ---
SMALL_SCALE = 1
BIG_SCALES = [2, 3]

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
    for r, (role, mask) in enumerate(zip(roles, masks, strict=False)):
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


def _union_bbox(patch_a: np.ndarray, patch_b: np.ndarray) -> tuple[int, int, int, int]:
    oa = _anchor_bbox(patch_a)
    ia = _anchor_bbox(patch_b)
    dr = oa[0] - ia[0]
    dc = oa[1] - ia[1]
    h, w = patch_a.shape
    return dr, dc, h + abs(dr), w + abs(dc)


def _build_instances() -> list[tuple[int, int, bool]]:
    big_scale_a = random.choice(BIG_SCALES)
    big_scale_b = random.choice(BIG_SCALES)
    instances = [
        (0, SMALL_SCALE, False),
        (1, SMALL_SCALE, False),
        (0, big_scale_a, True),
        (1, big_scale_b, True),
    ]
    random.shuffle(instances)
    instances.sort(key=lambda item: item[1], reverse=True)
    return instances


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_CHOICES)
    proto_a, proto_b = _make_prototypes(bg)
    protos = [proto_a, proto_b]
    instances = _build_instances()

    max_dim = max(max(protos[typ].shape) * scale for typ, scale, _swap in instances)
    n = random.randint(max(GRID_SIZE_MIN, max_dim + GRID_MARGIN_FOR_SPRITES), GRID_SIZE_MAX)
    out = np.full((n, n), bg, dtype=int)
    inp = np.full((n, n), bg, dtype=int)
    reserve = np.full((n, n), bg, dtype=int)

    for out_type, scale, will_swap in instances:
        output_patch = _scale(protos[out_type], scale)
        input_type = 1 - out_type if will_swap else out_type
        input_patch = _scale(protos[input_type], scale)
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

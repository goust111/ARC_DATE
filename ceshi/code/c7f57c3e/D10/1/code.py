# Variant of c7f57c3e
# Dimension: D10 — 精灵数量与分组基数 (multi_step)
# Change: 精灵数量不定，分为恰好两组，组间1:1或多:多配对... → 一组多数vs一组少数

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the D10 variant
# -----------------------------------
# a) 具体化与行为追踪
#    - 仍然只有两种多色精灵原型，仍然共享 color-1 锚点区，仍然允许 1-3 倍整数缩放。
#    - 关键变化不在“替换后长什么样”，而在“输入里如何看出谁给谁传颜色模式”：
#      一种原型出现 2-3 次，另一种原型只出现 1 次。少数组唯一精灵把自己的颜色模式复制给多数派所有精灵，
#      多数派的模式则回传给那个唯一精灵。
#    - 为了让这种一对多关系在输入中是可见且必要的，生成器强制：只有多数派会在最小尺度出现；
#      少数派唯一精灵只以放大版出现。于是“从最小面积精灵直接读出两个原型”这个旧机制失效。
#    - 小例子：输入里若 A 组有三个精灵，尺度为 1/2/3，B 组只有一个尺度 2 的精灵，
#      则输出把三个 A 都改成 B 的图案，把那个 B 改成 A 的图案。和原题不同的是，
#      solver 不能再依赖“最小面积中有 A 和 B 各一个”，必须先把所有精灵按锚点缩回基型再分组计数。
# b) 复杂度校准
#    - 只保留 2 个原型，避免自由度过大；只允许 2:1 或 3:1，保证多数组/少数组在 2-4 个样例内稳定可见。
#    - 仍保留 2-3 倍缩放和内部双色区域，避免视觉上退化成纯复制或纯计数题。
#    - 少数派只出现 1 个，但其图案会作用到多个对象上，因此一对多效果明显。
# c) 原 main() 中依赖旧属性的写法审查
#    - `base = [s for s, a in zip(...) if a == min_area]` 依赖“最小面积层同时包含两个原型”；变体中不成立。
#    - `assert len(uniq) == 2` 依赖“最小面积去重后刚好得到两类”；变体中最小面积通常只会看到多数派。
#    - 旧版通过最小原型的锚点尺寸估计所有 scale，这在变体里仍可保留，但前提改成：
#      最小锚点 bbox 必须来自那唯一的尺度 1 多数派样本，而不是来自两个原型各一个样本。
#    - 旧版用“把当前精灵和两个原型打分匹配”来决定其类别；变体里更稳妥的做法是：
#      先根据锚点推 scale，再把每个精灵缩回基型，直接按基型内容分组。
#    - 旧版逐个清空再逐个回贴在原任务中是无害的；这里仍可行，但为避免一对多替换时中间态污染评分，
#      新版先完成全部分组，再统一清空，再统一回贴。
# d) 放置合法性
#    - 仍使用原题的锚点对齐 union bbox 预留法，保证输入 patch 与输出 patch 都能完整落在同一预留区内。
#    - 精灵之间保持 >=2 的间隔和 >=1 的边距，且 union bbox 彼此不重叠，因此求解后的输出也不会互相覆盖。
#    - 生成器固定只有一个尺度 1 精灵，且它属于多数派；这保证 solver 的最小锚点推断唯一且稳定。
# e) 视觉可观察性
#    - 人类能直接看到“一种图案出现很多次，另一种只出现一次”，输出又把单个图案扩散给多数派。
#    - 不使用颜色 id 排序，也不依赖不可见的随机标签；分组完全由像素图案和出现次数决定。
# f) 可视化整洁度
#    - 不引入额外标记或遮罩，仍是分离摆放的精灵；视觉杂讯主要由原型自身承担，整体布局保持整洁。
# g) 颜色变化可推断性
#    - 每个原型仍只有 2 个可变色区，且伴随固定结构位置；观察输入输出时可以看出“整套颜色模式被复制”，
#      而不是独立像素重涂。


def _pick_anchor_bbox(arr: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    """Pick the bbox of the color-1 anchor region; the region is guaranteed to exist."""
    if np.any(arr == 1):
        ys, xs = np.where(arr == 1)
        return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())
    assert False, "sprite array has no color-1 pixels and should always have them"


def _scale_nn(arr: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def _reduce_nn(arr: np.ndarray, scale: int) -> np.ndarray:
    """Inverse of nearest-neighbor scaling for perfectly replicated blocks."""
    if scale <= 1:
        return arr.copy()
    h, w = arr.shape
    assert h % scale == 0 and w % scale == 0, "scaled sprite must divide evenly by its scale"
    reduced = arr[::scale, ::scale].copy()
    for r in range(reduced.shape[0]):
        for c in range(reduced.shape[1]):
            block = arr[r * scale : (r + 1) * scale, c * scale : (c + 1) * scale]
            assert np.all(block == reduced[r, c]), "sprite blocks must be constant under NN scaling"
    return reduced


def _extract_sprites(g: np.ndarray, bg: int) -> list[dict]:
    comps = components_mask(g != bg, connectivity=8)
    assert len(comps) >= 3, "grid must contain a 2:1 or 3:1 split of sprite components"
    sprites = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        sprites.append({"pts": pts, "bbox": (r0, c0, r1, c1), "arr": arr})
    return sprites


def _prepare_records(sprites: list[dict], bg: int) -> tuple[list[dict], dict[tuple[int, ...], dict], tuple[int, int]]:
    """Recover each sprite's scale and reduced prototype, then group identical prototypes."""
    anchor_sizes = []
    for s in sprites:
        ar0, ac0, ar1, ac1 = _pick_anchor_bbox(s["arr"], bg)
        anchor_sizes.append((ar1 - ar0 + 1, ac1 - ac0 + 1))
    base_anchor_h = min(h for h, _ in anchor_sizes)
    base_anchor_w = min(w for _, w in anchor_sizes)

    records = []
    groups: dict[tuple[int, ...], dict] = {}
    for s in sprites:
        r0, c0, _r1, _c1 = s["bbox"]
        arr = s["arr"]
        anc = _pick_anchor_bbox(arr, bg)
        ar0, ac0, ar1, ac1 = anc
        inst_anc_h = ar1 - ar0 + 1
        inst_anc_w = ac1 - ac0 + 1
        assert inst_anc_h % base_anchor_h == 0 and inst_anc_w % base_anchor_w == 0, "anchor scale must be integral"
        scale_h = inst_anc_h // base_anchor_h
        scale_w = inst_anc_w // base_anchor_w
        assert scale_h == scale_w, "anchor scaling must be isotropic"
        scale = int(scale_h)

        base_arr = _reduce_nn(arr, scale)
        key = tuple(base_arr.ravel().tolist())
        rec = {
            "pts": s["pts"],
            "bbox": s["bbox"],
            "arr": arr,
            "anchor": anc,
            "scale": scale,
            "base_arr": base_arr,
            "key": key,
            "anchor_top_left": (r0 + ar0, c0 + ac0),
        }
        records.append(rec)
        if key not in groups:
            groups[key] = {"proto": base_arr, "records": []}
        groups[key]["records"].append(rec)

    return records, groups, (base_anchor_h, base_anchor_w)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the singleton group's pattern onto every majority sprite, and send the majority pattern back to the singleton.

    Input structure:
    - bg is the dominant color. Exactly two sprite prototypes are present, but their counts are asymmetric: one prototype appears 2-3 times and the other appears exactly once.
    - Every sprite is an integer-scaled nearest-neighbor copy of one prototype and contains a color-1 anchor region. Only the majority prototype is guaranteed to appear at the smallest scale.

    Rule:
    - Infer the common base anchor size from the smallest anchor bbox, recover each sprite's scale, and shrink every sprite back to its base prototype.
    - Group identical reduced prototypes, identify the majority group and the singleton group by occurrence count, then swap the two group patterns.
    - Repaint every sprite with the opposite group's prototype at its own original scale, aligned so the color-1 anchor stays fixed in place.

    Key insight:
    - The variant is no longer solved by reading both prototypes directly from the smallest whole sprites, because only the majority group appears there.
    - The stable color-1 anchor reveals each sprite's scale, so even the lone minority sprite can be reduced to its base form and then broadcast to all majority copies.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    sprites = _extract_sprites(g, bg)
    records, groups, _base_anchor = _prepare_records(sprites, bg)
    assert len(groups) == 2, "grid must contain exactly 2 prototype families"

    ordered = sorted(groups.items(), key=lambda item: len(item[1]["records"]))
    singleton_key, singleton_group = ordered[0]
    majority_key, majority_group = ordered[1]
    assert len(singleton_group["records"]) == 1, "one prototype family must be the singleton donor"
    assert len(majority_group["records"]) >= 2, "the other prototype family must be the majority recipient"

    replacement_by_key = {
        majority_key: singleton_group["proto"],
        singleton_key: majority_group["proto"],
    }

    out = g.copy()
    for rec in records:
        for rr, cc in rec["pts"]:
            out[rr, cc] = bg

    for rec in records:
        r0, c0, _r1, _c1 = rec["bbox"]
        ar0, ac0, _ar1, _ac1 = rec["anchor"]
        scale = rec["scale"]
        proto = replacement_by_key[rec["key"]]
        scaled_proto = _scale_nn(proto, scale)
        pr0, pc0, _pr1, _pc1 = _pick_anchor_bbox(proto, bg)
        top = (r0 + ar0) - pr0 * scale
        left = (c0 + ac0) - pc0 * scale

        ph, pw = scaled_proto.shape
        rr0 = max(0, top)
        cc0 = max(0, left)
        rr1 = min(h, top + ph)
        cc1 = min(w, left + pw)
        sub_p = scaled_proto[(rr0 - top) : (rr1 - top), (cc0 - left) : (cc1 - left)]
        mask = sub_p != bg
        out_slice = out[rr0:rr1, cc0:cc1]
        out_slice[mask] = sub_p[mask]
        out[rr0:rr1, cc0:cc1] = out_slice

    return out


# --------------- generator (o5x variant) ---------------

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
MAJORITY_COUNTS = [2, 3]
BIG_SCALES = [2, 3]

# --- CFG: placement ---
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2
PLACEMENT_CONNECTIVITY = 8

# --- CFG: generation retries ---
MAX_RETRIES = 300

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
    ib = _anchor_bbox(patch_b)
    dr = oa[0] - ib[0]
    dc = oa[1] - ib[1]
    top = min(0, dr)
    left = min(0, dc)
    bottom = max(patch_a.shape[0], dr + patch_b.shape[0])
    right = max(patch_a.shape[1], dc + patch_b.shape[1])
    return top, left, bottom - top, right - left


def _smallest_area_unique_count(inp: np.ndarray) -> int:
    bg = get_bg(inp)
    sprites = _extract_sprites(inp, bg)
    areas = []
    for s in sprites:
        r0, c0, r1, c1 = s["bbox"]
        areas.append((r1 - r0 + 1) * (c1 - c0 + 1))
    min_area = min(areas)
    keys = set()
    for s, area in zip(sprites, areas, strict=False):
        if area == min_area:
            keys.add(tuple(s["arr"].ravel().tolist()))
    return len(keys)


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_CHOICES)
    proto_a, proto_b = _make_prototypes(bg)
    protos = [proto_a, proto_b]

    majority_type = random.randrange(2)
    minority_type = 1 - majority_type
    majority_count = random.choice(MAJORITY_COUNTS)

    instances = [(majority_type, 1)]
    for _ in range(majority_count - 1):
        instances.append((majority_type, random.choice(BIG_SCALES)))
    instances.append((minority_type, random.choice(BIG_SCALES)))
    random.shuffle(instances)
    instances.sort(key=lambda item: item[1], reverse=True)

    max_dim = max(max(protos[typ].shape) * scale for typ, scale in instances)
    n = random.randint(max(GRID_SIZE_MIN, max_dim + GRID_MARGIN_FOR_SPRITES), GRID_SIZE_MAX)
    inp = np.full((n, n), bg, dtype=int)
    out = np.full((n, n), bg, dtype=int)
    reserve = np.full((n, n), bg, dtype=int)

    for input_type, scale in instances:
        input_patch = _scale(protos[input_type], scale)
        output_patch = _scale(protos[1 - input_type], scale)
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

        out_r = ur - dr
        out_c = uc - dc
        in_r = out_r + dr
        in_c = out_c + dc

        paste(reserve, marker, ur, uc)
        paste(out, output_patch, out_r, out_c, transparent=bg)
        paste(inp, input_patch, in_r, in_c, transparent=bg)

    if _smallest_area_unique_count(inp) != 1:
        return None

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
    raise RuntimeError("failed to generate a valid D10 c7f57c3e input")

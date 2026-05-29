# Combo variant of c7f57c3e (index 27)
# combo_spec: D9/3+D11/1
# Name: 条件性交换 + 背景色随主色组变化
#
# Design analysis
# a) 具体化与行为追踪
#    输入仍是两类共享 color-1 锚点的多色精灵，最小 bbox 的两个 scale-1 精灵直接给出 proto_A / proto_B。
#    这里先做 D9/3 的参与判定：只有 bbox 面积严格大于最小面积的放大精灵会交换；两个最小精灵保持原样。
#    然后叠加 D11/1：不是看全部精灵，而是只统计这些“会交换”的参与者在输入中属于哪一组。若参与者多数是 A，
#    则输出背景改成 B 的关联色；若多数是 B，则背景改成 A 的关联色。例子：小 A、小 B、不参与；两个大 A 参与，
#    那么输出里两个大 A 都变成大 B，且整张背景一起变成 B 的关联色。原题是“全员交换且背景不变”，这里则是
#    “先筛参与者，再用参与者多数派控制全局背景”。
# b) 复杂度校准
#    参与条件只使用一个人眼可见的离散特征：是否大于最小 bbox。背景判断也只使用两档多数派：A 多 / B 多。
#    生成器固定放 4 个精灵：2 个最小原型样本 + 2 个放大参与者，并强制这 2 个参与者同属一个输入组，
#    从而避免平局与背景规则失效，同时保证每个样例里都有重复证据支撑“谁参与、谁决定背景”。
# c) 隐式假设审查
#    原 solver 默认每个对象都 `repl_idx = 1 - best_idx`，融合后这个前提失效，所以输出不能从 `g.copy()` 出发逐个改，
#    而要先判定参与者和输出背景，再从整张新背景重建所有精灵。否则未参与的小精灵会被误换，且背景无法整体替换。
#    原代码还隐含“背景色不等于任何重要前景色”。这里故意打破这一点：背景会变成对侧原型的关联色，因此必须让
#    关联色只占该原型较小的强调区域，避免整只精灵大面积淹没。生成器把每个原型的关联色固定为较稀少的 role-3 颜色，
#    solver 则通过“非 bg / 非 1 颜色里最少见的那个”恢复该关联色。
# d) 放置合法性
#    继续采用 output-first。先确定输出里的小精灵保持原型不动、两个大精灵交换到对侧原型，再根据参与者输入多数派
#    决定输出背景色。放置时仍对输入 patch 与输出 patch 做锚点对齐后的并集预留，确保输入与输出都不重叠、不越界。
# e) 视觉可观察性
#    人类可以直接看到两类线索同时出现：只有放大精灵发生模式互换，而整张背景颜色也跟着换成“交换后占优那一侧”的
#    关联色。这个判断完全基于可见的尺寸、原型类别和全局背景，不依赖颜色 id 的数值大小或任何隐藏顺序。
# f) 可视化整洁度
#    仍只放 4 个分离精灵，保持 >=2 的间隔；背景虽然改色，但不再额外叠加标记。关联色只占原型中的小区域，
#    所以即使背景与该颜色相同，精灵仍保留 anchor 和主体色，整体轮廓不会乱成一片。
# g) 变化的可推断性
#    每个原型都有两种可变色，其中一种主体色覆盖更大，另一种关联色覆盖更小且两组互不共享。于是观察者既能通过
#    “哪类大精灵参与交换”推断背景跟谁走，也能从背景恰好变成对侧原型的小强调色，看出这是组关联色而非随机换色。
# h) 新的 main() 隐式假设与 generator 保证
#    `main()` 新假设包括：最小面积层恰好给出两种原型；所有更大对象仍是它们的整数倍缩放；参与者多数派不平局时
#    背景才切换；每个原型的关联色在其变量色中出现次数最少。`generate_input()` 通过固定 2 小 2 大、让 2 个大对象
#    同属一个输入组、并给 role-3 只分配 1 行来保证这些前提，再用 oracle 校验 `main(inp) == out`。

import random
from collections import Counter

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


def _pick_anchor_bbox(arr: np.ndarray, bg: int) -> tuple[int, int, int, int]:
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


def _infer_scale(
    arr: np.ndarray, base_anchor_h: int, base_anchor_w: int, bg: int
) -> tuple[int, tuple[int, int, int, int]]:
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
    for idx, proto in enumerate(protos):
        scaled = _scale_nn(proto, scale)
        pr0, pc0, _pr1, _pc1 = _pick_anchor_bbox(proto, bg)
        top = (r0 + ar0) - (pr0 * scale)
        left = (c0 + ac0) - (pc0 * scale)

        ph, pw = scaled.shape
        rr0 = max(0, top)
        cc0 = max(0, left)
        rr1 = min(h, top + ph)
        cc1 = min(w, left + pw)
        if rr0 >= rr1 or cc0 >= cc1:
            continue
        sub_g = g[rr0:rr1, cc0:cc1]
        sub_p = scaled[(rr0 - top) : (rr1 - top), (cc0 - left) : (cc1 - left)]
        m = sub_p != bg
        score = int(np.sum(sub_g[m] == sub_p[m])) if np.any(m) else 0
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def _signature_color(arr: np.ndarray, bg: int) -> int:
    colors = [int(v) for v in arr.ravel().tolist() if v not in (bg, 1)]
    counts = Counter(colors)
    assert len(counts) == 2, "prototype must use exactly two variable colors"
    min_count = min(counts.values())
    for v in arr.ravel().tolist():
        color = int(v)
        if color in counts and counts[color] == min_count:
            return color
    assert False, "signature color lookup failed"


def _paste_aligned(
    out: np.ndarray,
    proto: np.ndarray,
    bbox: tuple[int, int, int, int],
    anc: tuple[int, int, int, int],
    scale: int,
    bg: int,
) -> None:
    h, w = out.shape
    r0, c0, _r1, _c1 = bbox
    ar0, ac0, _ar1, _ac1 = anc
    scaled = _scale_nn(proto, scale)
    pr0, pc0, _pr1, _pc1 = _pick_anchor_bbox(proto, bg)
    top = (r0 + ar0) - (pr0 * scale)
    left = (c0 + ac0) - (pc0 * scale)

    ph, pw = scaled.shape
    rr0 = max(0, top)
    cc0 = max(0, left)
    rr1 = min(h, top + ph)
    cc1 = min(w, left + pw)
    sub_p = scaled[(rr0 - top) : (rr1 - top), (cc0 - left) : (cc1 - left)]
    m = sub_p != bg
    out_slice = out[rr0:rr1, cc0:cc1]
    out_slice[m] = sub_p[m]
    out[rr0:rr1, cc0:cc1] = out_slice


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap only the enlarged sprites, then recolor the whole background from the participating majority.

    Input structure:
    - The most common color is the input background. Separate components are scaled copies of exactly two prototype sprites.
    - The smallest bbox-area sprites expose both prototypes directly, while any larger sprites are integer enlargements of one of them.

    Rule:
    - Recover the two prototypes from the minimum-area sprites and treat only larger sprites as swap participants.
    - For every sprite, infer its prototype identity and integer scale from the color-1 anchor bbox. Non-participants keep their own prototype; participants switch to the opposite one at the same anchored placement.
    - Count only the participating sprites by input prototype. If A-participants are the majority, rebuild the entire output on B's associated color; if B-participants are the majority, rebuild it on A's associated color. A tie leaves the background unchanged.

    Key insight: The smallest sprites serve two jobs at once: they reveal the base prototypes and define who is exempt. Once that participation set is known, it controls both the local swaps and the global background color, so the output must be reconstructed from scratch rather than edited in place.
    """
    g = np.asarray(grid, dtype=int)
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
    base = [s for s, area in zip(sprites, areas, strict=False) if area == min_area]
    uniq: dict[tuple[int, ...], np.ndarray] = {}
    for sprite in base:
        key = tuple(sprite["arr"].ravel().tolist())
        if key not in uniq:
            uniq[key] = sprite["arr"]
    assert len(uniq) == 2, "smallest-area sprites must yield exactly 2 distinct prototype patterns"

    protos = list(uniq.values())[:2]
    anc0 = _pick_anchor_bbox(protos[0], bg)
    anc1 = _pick_anchor_bbox(protos[1], bg)
    base_anc_h0 = anc0[2] - anc0[0] + 1
    base_anc_w0 = anc0[3] - anc0[1] + 1
    base_anc_h1 = anc1[2] - anc1[0] + 1
    base_anc_w1 = anc1[3] - anc1[1] + 1
    assert (base_anc_h0, base_anc_w0) == (base_anc_h1, base_anc_w1), "prototype anchors must share a size"

    signature_colors = [_signature_color(proto, bg) for proto in protos]
    classified = []
    participant_counts = [0, 0]
    for sprite, area in zip(sprites, areas, strict=False):
        scale, anc = _infer_scale(sprite["arr"], base_anc_h0, base_anc_w0, bg)
        proto_idx = _match_prototype(g, sprite["bbox"], protos, scale, anc, bg)
        participates = area > min_area
        if participates:
            participant_counts[proto_idx] += 1
        classified.append(
            {
                "bbox": sprite["bbox"],
                "anc": anc,
                "scale": scale,
                "proto_idx": proto_idx,
                "participates": participates,
            }
        )

    if participant_counts[0] > participant_counts[1]:
        out_bg = signature_colors[1]
    elif participant_counts[1] > participant_counts[0]:
        out_bg = signature_colors[0]
    else:
        out_bg = bg

    out = np.full_like(g, out_bg)
    for info in classified:
        target_idx = 1 - info["proto_idx"] if info["participates"] else info["proto_idx"]
        _paste_aligned(out, protos[target_idx], info["bbox"], info["anc"], info["scale"], bg)
    return out


# --------------- generator (o5x) ---------------

# --- CFG: canvas ---
GRID_SIZE_MIN = 18
GRID_SIZE_MAX = 28
GRID_MARGIN_FOR_SPRITES = 6

# --- CFG: prototypes ---
PROTO_SHAPES = [(4, 4), (5, 4), (5, 5)]
BG_CHOICES = [0, 2, 3, 4, 5, 6, 7, 8, 9]

# --- CFG: row-mask weights ---
ANCHOR_ROW_WEIGHTS = [("center", 0.45), ("outer", 0.30), ("full", 0.25)]
BODY_ROW_WEIGHTS = [("center", 0.35), ("outer", 0.15), ("full", 0.50)]
SIGNATURE_ROW_WEIGHTS = [("center", 0.80), ("outer", 0.20)]
OUTER_FIXUP_PROB = 0.7

# --- CFG: sprite instances ---
BIG_SCALES = [2, 3]

# --- CFG: placement ---
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2
PLACEMENT_CONNECTIVITY = 8

# --- CFG: generation retries ---
MAX_RETRIES = 300

_ROW_MASKS = {
    "center": {4: [0, 1, 1, 0], 5: [0, 1, 1, 1, 0]},
    "outer": {4: [1, 0, 0, 1], 5: [1, 0, 0, 0, 1]},
    "full": {4: [1, 1, 1, 1], 5: [1, 1, 1, 1, 1]},
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
    roles = [1] + [2] * (h - 2) + [3]
    random.shuffle(roles)

    weights_by_role = {
        1: ANCHOR_ROW_WEIGHTS,
        2: BODY_ROW_WEIGHTS,
        3: SIGNATURE_ROW_WEIGHTS,
    }
    masks = []
    for role in roles:
        kinds, weights = zip(*weights_by_role[role])
        masks.append(_row_mask(random.choices(kinds, weights=weights)[0], w))

    if not any(mask[0] or mask[-1] for mask in masks):
        i = random.randrange(h)
        if roles[i] == 1 and random.random() < OUTER_FIXUP_PROB:
            masks[i] = _row_mask("outer", w)
        else:
            masks[i] = _row_mask("full", w)

    if all(all(mask) for mask in masks):
        candidates = [i for i, role in enumerate(roles) if role != 1]
        i = random.choice(candidates)
        masks[i] = _row_mask("center", w)

    labels = np.zeros((h, w), dtype=int)
    for r, (role, mask) in enumerate(zip(roles, masks, strict=False)):
        for c, bit in enumerate(mask):
            if bit:
                labels[r, c] = role
    return labels


def _make_prototypes(bg: int) -> tuple[np.ndarray, np.ndarray]:
    labels = _make_labels()
    pool = [c for c in range(10) if c not in (bg, 1)]
    a_body, a_sig, b_body, b_sig = random.sample(pool, 4)

    def _colorize(lab: np.ndarray, body: int, sig: int) -> np.ndarray:
        proto = np.full(lab.shape, bg, dtype=int)
        proto[lab == 1] = 1
        proto[lab == 2] = body
        proto[lab == 3] = sig
        return proto

    proto_a = _colorize(labels, a_body, a_sig)
    proto_b = _colorize(np.flipud(labels), b_body, b_sig)
    return proto_a, proto_b


def _union_bbox(patch_a: np.ndarray, patch_b: np.ndarray) -> tuple[int, int, int, int]:
    out_anchor = _anchor_bbox(patch_a)
    in_anchor = _anchor_bbox(patch_b)
    dr = out_anchor[0] - in_anchor[0]
    dc = out_anchor[1] - in_anchor[1]
    h = max(patch_a.shape[0], patch_b.shape[0])
    w = max(patch_a.shape[1], patch_b.shape[1])
    return dr, dc, h + abs(dr), w + abs(dc)


def _build_instances() -> tuple[list[dict[str, int]], int]:
    majority_input_type = random.randrange(2)
    big_scales = BIG_SCALES.copy()
    random.shuffle(big_scales)
    instances = [
        {"input_type": 0, "output_type": 0, "scale": 1},
        {"input_type": 1, "output_type": 1, "scale": 1},
        {"input_type": majority_input_type, "output_type": 1 - majority_input_type, "scale": big_scales[0]},
        {"input_type": majority_input_type, "output_type": 1 - majority_input_type, "scale": big_scales[1]},
    ]
    random.shuffle(instances)
    instances.sort(key=lambda item: item["scale"], reverse=True)
    return instances, majority_input_type


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_CHOICES)
    proto_a, proto_b = _make_prototypes(bg)
    protos = [proto_a, proto_b]
    sig_colors = [_signature_color(proto_a, bg), _signature_color(proto_b, bg)]
    instances, majority_input_type = _build_instances()
    out_bg = sig_colors[1 - majority_input_type]

    max_dim = max(max(protos[item["input_type"]].shape) * item["scale"] for item in instances)
    n = random.randint(max(GRID_SIZE_MIN, max_dim + GRID_MARGIN_FOR_SPRITES), GRID_SIZE_MAX)
    inp = np.full((n, n), bg, dtype=int)
    out = np.full((n, n), out_bg, dtype=int)
    reserve = np.full((n, n), bg, dtype=int)

    for item in instances:
        output_patch = _scale(protos[item["output_type"]], item["scale"])
        input_patch = _scale(protos[item["input_type"]], item["scale"])
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
        in_r = out_r + dr
        in_c = out_c + dc

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
    raise RuntimeError("failed to generate a valid c7f57c3e combo input")

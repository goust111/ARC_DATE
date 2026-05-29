# Variant of c7f57c3e
# Dimension: D10 — 精灵数量与分组基数 (multi_step)
# Change: 精灵数量不定，分为恰好两组，组间1:1或多:多配对... → 三组颜色模式三向交换

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis
# ===============
# design_analysis:
#   source_task: none
#   concretization:
#     objects: |
#       画面里仍然只有若干互不接触的多色精灵，每个精灵都是某个小 prototype 的整数倍放大版。
#       与原题不同的是，prototype 不再只有 2 类，而是恰好 3 类。三类 prototype 的几何轮廓、
#       背景孔洞和 color-1 锚区完全相同，只在两块“变量区域”的配色上不同。
#       变量颜色按一个可见的有向三环出现：P0=(c0,c1), P1=(c1,c2), P2=(c2,c0)。
#       因此输入里会看到三种重复出现的双色模式；输出把每个精灵推进到下一种模式。
#     tracked_example: |
#       设三个最小精灵分别是 (2,6), (6,8), (8,2) 三种颜色对，另有一个 scale=2 的大精灵属于 (8,2)。
#       原逻辑: 只有两类时，大精灵会切到“另一类”。
#       新逻辑: solver 先从最小精灵恢复三类，再从颜色对的有向关系读出环
#       (2,6)->(6,8)->(8,2)->(2,6)。于是 scale=2 的 (8,2) 大精灵必须变成 (2,6)，
#       其锚框和尺度不变，只替换颜色模式。
#   complexity_calibration:
#     kept:
#       - 仍是 3x4 / 4x3 / 4x4 小精灵，带内部孔洞、color-1 锚区、整数倍缩放、非重叠散布。
#       - 仍使用 output-first 生成并用 solver 回验。
#     changed:
#       - 最小面积样本必须能恢复 3 个 prototype，而不是 2 个。
#       - 颜色模式不是二选一切换，而是三向有序环换。
#     anti_simplification: |
#       三类 prototype 共享三种变量色而不是各自完全独立配色，这样人类能从少量样例看出“颜色对沿环推进”，
#       同时又不会退化成随便换成另一组颜色。
#   implicit_assumption_review:
#     old_dependencies:
#       - 原 `len(uniq) == 2` 依赖最小面积实例里只需恢复 2 类 prototype。
#       - 原 `repl_idx = 1 - best_idx` 依赖类别集合无方向、无顺序。
#       - 原 solver 对 prototype 顺序不敏感；三环版本则必须从可见特征恢复一个有向顺序。
#     new_resolution: |
#       新 solver 仍从最小面积实例抽 prototype，但要求最小面积实例的去重结果恰好为 3。
#       prototype 的顺序不再来自扫描顺序，而是来自“变量颜色对”的有向拼接关系：
#       若某类的第二变量色等于另一类的第一变量色，则它指向下一类。
#   placement_legality:
#     generator_guarantees:
#       - 总精灵数为 4 或 5：固定放入 3 个 scale=1 的基座精灵（每类各一个），再放 1 个 scale=2/3 大精灵，
#         以及可选 1 个额外 scale=1 精灵。
#       - 所有精灵通过 `random_free_location` 在带 padding 的 reserve 画布上放置，互不重叠且不越界。
#       - 输入与输出 prototype 结构相同，所以 anchor 对齐后 bbox 保持一致，solver 的替换不会覆盖其他对象。
#   visual_observability:
#     human_view: |
#       观察者能直接看到三种反复出现的双色模式，例如 (红,粉)、(粉,青)、(青,红)。
#       输入到输出时，每个精灵都沿同一方向前进一步，而不是两类互换；三种模式会在同一题里同时出现，
#       这使“三环”规律在 2-3 个样例里可见。
#   visual_tidiness:
#     choices:
#       - 保持对象间至少 2 格间距，避免相邻精灵视觉粘连。
#       - 变量区数量仍然只有 2 块，总颜色数控制在 bg + 1 + 3cycle 之内，避免画面过杂。
#   color_inference:
#     constraints: |
#       颜色环的顺序完全由像素位置中的“第一变量区/第二变量区”给出，不依赖颜色 id 排序。
#       generator 只随机挑选三种 distinct 颜色作为 cycle colors；solver 通过区域出现顺序恢复它们的有向关系。
#   solver_assumptions_ensured_by_generator:
#     - 每个 prototype 恰有两种变量色，且三类 prototype 的变量色对组成一个长度为 3 的简单有向环。
#     - 所有 prototype 的 color-1 锚框尺寸一致，因此可以从锚框尺寸推回整数 scale。
#     - 最小面积组件里至少包含三类各一个，因此 `main()` 能稳定恢复完整的 prototype 集合。


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


def _ordered_variable_pair(arr: np.ndarray, bg: int) -> tuple[int, int]:
    """Return the two non-bg non-anchor colors ordered by first row-major appearance."""
    colors = [int(c) for c in np.unique(arr) if int(c) not in (bg, 1)]
    assert len(colors) == 2, "each prototype must have exactly two variable colors"
    first_hits = []
    for col in colors:
        flat = np.flatnonzero(arr == col)
        assert flat.size > 0
        first_hits.append((int(flat[0]), col))
    first_hits.sort()
    return int(first_hits[0][1]), int(first_hits[1][1])


def _recover_cycle_order(base_protos: list[np.ndarray], bg: int) -> list[np.ndarray]:
    """Recover the directed 3-cycle order from the observable variable-color pairs."""
    pairs = [_ordered_variable_pair(p, bg) for p in base_protos]
    first_to_idx: dict[int, int] = {}
    for idx, pair in enumerate(pairs):
        assert pair[0] not in first_to_idx, "cycle colors must start distinct prototypes"
        first_to_idx[pair[0]] = idx
    assert len(first_to_idx) == 3, "expected exactly three distinct cycle start colors"

    next_idx: dict[int, int] = {}
    for idx, pair in enumerate(pairs):
        assert pair[1] in first_to_idx, "prototype pairs must chain into a cycle"
        next_idx[idx] = first_to_idx[pair[1]]

    order = [0]
    while len(order) < 3:
        nxt = next_idx[order[-1]]
        assert nxt not in order, "prototype chain must visit three distinct groups"
        order.append(nxt)
    assert next_idx[order[-1]] == order[0], "prototype chain must close into a 3-cycle"
    return [base_protos[i] for i in order]


def _classify_sprite(arr: np.ndarray, protos: list[np.ndarray], bg: int, scale: int) -> int:
    """Return which prototype the sprite matches."""
    exact = []
    for idx, proto in enumerate(protos):
        sp = _scale_nn(proto, scale)
        if sp.shape == arr.shape and np.array_equal(sp, arr):
            exact.append(idx)
    if len(exact) == 1:
        return exact[0]

    best_idx = 0
    best_score = -1
    for idx, proto in enumerate(protos):
        sp = _scale_nn(proto, scale)
        if sp.shape != arr.shape:
            continue
        score = int(np.sum(arr == sp))
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def main(grid: np.ndarray) -> np.ndarray:
    """
    Advance every sprite to the next color-pattern prototype in a 3-cycle while preserving scale and placement.

    Input structure:
    - The background is the most common color. Exactly three distinct small sprite prototypes appear, each sharing the same shape and color-1 anchor region.
    - Every visible sprite is an integer-scaled copy of one of those prototypes; the prototypes differ only by which two variable colors occupy the two internal variable zones.

    Rule:
    - Recover the three base prototypes from the smallest sprite instances, then infer their cyclic order from the observable variable-color pairs.
    - For each sprite, detect its prototype group and integer scale, and replace it with the next prototype in the cycle A->B, B->C, C->A.
    - The replacement keeps the same scale and is aligned by the color-1 anchor bbox so the sprite stays in the same location.

    Key insight: The smallest sprites expose all three color patterns, and those patterns themselves encode a directed cycle because one prototype's second variable color becomes the next prototype's first variable color. Once that order is recovered, each larger scaled sprite can be advanced one step while the anchor region locks its placement.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=8)
    assert len(comps) >= 3, "grid must contain at least 3 sprite components"

    sprites = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        sprites.append({"pts": pts, "bbox": (r0, c0, r1, c1), "arr": arr})

    areas = [int((s["bbox"][2] - s["bbox"][0] + 1) * (s["bbox"][3] - s["bbox"][1] + 1)) for s in sprites]
    min_area = min(areas)
    base = [s for s, a in zip(sprites, areas, strict=False) if a == min_area]
    uniq: dict[tuple[int, ...], np.ndarray] = {}
    for s in base:
        key = tuple(s["arr"].ravel().tolist())
        if key not in uniq:
            uniq[key] = s["arr"]
    assert len(uniq) == 3, "smallest-area sprites must yield exactly 3 distinct prototype patterns"

    protos = _recover_cycle_order(list(uniq.values()), bg)

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

        best_idx = _classify_sprite(arr, protos, bg, scale)
        repl_idx = (best_idx + 1) % 3

        for rr, cc in s["pts"]:
            out[rr, cc] = bg

        proto = protos[repl_idx]
        sp = _scale_nn(proto, scale)
        panc = _pick_anchor_bbox(proto, bg)
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
SPRITE_COUNTS = [4, 5]
BIG_SCALES = [2, 3]

# --- CFG: placement ---
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2
PLACEMENT_CONNECTIVITY = 8

# --- CFG: generation retries ---
MAX_RETRIES = 400

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
    """Build an (h, w) label grid with roles 1 (anchor) and two variable zones."""
    for _ in range(200):
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

        if len(components_mask(labels != 0, connectivity=8)) != 1:
            continue
        if not np.any(labels == 2) or not np.any(labels == 3):
            continue
        first2 = int(np.flatnonzero(labels == 2)[0])
        first3 = int(np.flatnonzero(labels == 3)[0])
        if first2 == first3:
            continue
        return labels
    raise RuntimeError("failed to build connected labels")


def _ordered_variable_labels(labels: np.ndarray) -> tuple[int, int]:
    first2 = int(np.flatnonzero(labels == 2)[0])
    first3 = int(np.flatnonzero(labels == 3)[0])
    return (2, 3) if first2 < first3 else (3, 2)


def _make_prototypes(bg: int) -> list[np.ndarray]:
    """Create three prototypes whose variable-color pairs form a directed 3-cycle."""
    labels = _make_labels()
    zone_a, zone_b = _ordered_variable_labels(labels)
    pool = [c for c in range(10) if c not in (bg, 1)]
    cycle_cols = random.sample(pool, 3)

    def _colorize(cols: tuple[int, int]) -> np.ndarray:
        proto = np.full(labels.shape, bg, dtype=int)
        proto[labels == 1] = 1
        proto[labels == zone_a] = cols[0]
        proto[labels == zone_b] = cols[1]
        return proto

    return [
        _colorize((cycle_cols[0], cycle_cols[1])),
        _colorize((cycle_cols[1], cycle_cols[2])),
        _colorize((cycle_cols[2], cycle_cols[0])),
    ]


def _union_bbox(patch_a: np.ndarray, patch_b: np.ndarray) -> tuple[int, int, int, int]:
    oa = _anchor_bbox(patch_a)
    ia = _anchor_bbox(patch_b)
    dr = oa[0] - ia[0]
    dc = oa[1] - ia[1]
    h = max(patch_a.shape[0], patch_b.shape[0])
    w = max(patch_a.shape[1], patch_b.shape[1])
    return dr, dc, h + abs(dr), w + abs(dc)


def _build_instances() -> list[tuple[int, int]]:
    count = random.choice(SPRITE_COUNTS)
    big_scale = random.choice(BIG_SCALES)
    instances = [(0, 1), (1, 1), (2, 1), (random.randrange(3), big_scale)]
    if count == 5:
        instances.append((random.randrange(3), 1))
    random.shuffle(instances)
    instances.sort(key=lambda item: item[1], reverse=True)
    return instances


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_CHOICES)
    protos = _make_prototypes(bg)
    instances = _build_instances()

    max_dim = max(max(protos[typ].shape) * scale for typ, scale in instances)
    n = random.randint(max(GRID_SIZE_MIN, max_dim + GRID_MARGIN_FOR_SPRITES), GRID_SIZE_MAX)
    out = np.full((n, n), bg, dtype=int)
    inp = np.full((n, n), bg, dtype=int)
    reserve = np.full((n, n), bg, dtype=int)

    for out_type, scale in instances:
        output_patch = _scale(protos[out_type], scale)
        input_patch = _scale(protos[(out_type - 1) % 3], scale)
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
    raise RuntimeError("failed to generate a valid c7f57c3e D10 input")

# Variant of c7f57c3e
# Dimension: D3 — 位置重叠约束 (match_cond)
# Change: 配对后最大化精灵的位置重叠（maximize position overlap），交换后精灵尽量保持原位... → 位置由尺度编码
#
# Design analysis:
# spec:
#   preserved_structure:
#     - 背景为单色，网格里仍然只有两种多色原型，精灵仍是这两种原型的整数尺度放大版。
#     - 颜色 1 仍然是稳定锚点区，因此可以继续用它来推断尺度并区分原型。
#   d3_change:
#     - 原题把每个精灵换成另一种原型后，继续贴回自己的原位置。
#     - 本变体改成“槽位重排”：输入中的三个精灵先定义出三个水平槽位或三个垂直槽位，
#       输出时把“换型后的精灵”按尺度从小到大分配到这些槽位里。
#     - 因而位置不再由局部重叠对齐决定，而由全局排序规则决定。
#   behavior_trace:
#     - 例：输入横向三个槽位从左到右分别是 [scale2 的 B, scale1 的 A, scale1 的 B]。
#     - 原题输出会得到 [scale2 的 A, scale1 的 B, scale1 的 A]，仍在原槽位。
#     - 本题输出改为 [scale1 的 B, scale1 的 A, scale2 的 A]；大精灵被移到最右侧槽位。
#   complexity_calibration:
#     - 为了让“按大小排序”在 2-3 个样例中肉眼可见，生成器固定为 3 个精灵，尺度多重集固定为 [1, 1, 2]。
#     - 这样最小尺度组件依然恰好给出两种原型；同时又有一个更大的精灵提供明确的排序对比。
#     - 原题允许 scale=3、4 个精灵，但那样若每个槽位都必须能容纳最大精灵，在 28x28 内很容易越界，
#       所以这里把最大尺度收紧到 2，这是 D3 改动的必要配套，不是额外规则创新。
#   hidden_assumptions_review:
#     - 原 solver 的“按锚点重叠打分并贴回原位”依赖旧前提：输出目标位置就是当前精灵附近。
#     - 该前提在本变体失效，所以新的 solver 不能再按每个精灵独立回贴，而要先抽取全部精灵、排序，再统一回放。
#     - 新 solver 依赖的新前提是：三个精灵共线排成一排（横向或纵向），并且每个槽位都能容纳最大尺度精灵。
#       生成器会显式保证这一点。
#   legality:
#     - 生成器先按最大可能 bbox 预留三个槽位，槽位之间留 1 格间隔，所有精灵都以槽位左上角对齐放置。
#     - 因此任何 scale1/scale2 的换型结果都能安全落入任一槽位，不重叠、不越界。
#   observability:
#     - 输入顺序被强制打乱，保证大精灵不在最终应在的末槽位，因此输出一定发生位置重排。
#     - 排序只看可见尺度，不使用颜色 id、随机隐藏标记或不可见元数据。
#   neatness_and_color_inference:
#     - 每个槽位只放一个精灵，没有覆盖层；颜色变化仍然只是“换到另一种原型”，与原题保持一致，
#       解题者只需额外观察“大小顺序控制位置”这一条新规律。

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


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


def _row_mask(kind: str, w: int) -> list[int]:
    return list(_ROW_MASKS[kind][w])


def _anchor_bbox(arr: np.ndarray) -> tuple[int, int, int, int]:
    rs, cs = np.where(arr == 1)
    return int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())


def _scale(arr: np.ndarray, k: int) -> np.ndarray:
    if k == 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, k, axis=0), k, axis=1)


def _slot_offsets(total: int, count: int, span: int, border: int, gap: int) -> list[int]:
    used = 2 * border + count * span + (count - 1) * gap
    if used > total:
        raise ValueError("slots do not fit")
    extra = total - used
    gaps = [0] * (count + 1)
    for _ in range(extra):
        gaps[random.randrange(count + 1)] += 1
    out = []
    cur = border + gaps[0]
    for i in range(count):
        out.append(cur)
        cur += span
        if i < count - 1:
            cur += gap + gaps[i + 1]
    return out


def _sprite_center_key(sprite: dict[str, object]) -> tuple[int, int]:
    r0, c0, r1, c1 = sprite["bbox"]
    return int(r0 + r1), int(c0 + c1)


def _match_sprite(arr: np.ndarray, protos: list[np.ndarray], scale: int) -> int:
    for idx, proto in enumerate(protos):
        if np.array_equal(arr, _scale_nn(proto, scale)):
            return idx
    raise AssertionError("sprite does not match either prototype at inferred scale")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap every sprite to the opposite prototype, then reorder the swapped sprites by scale across the existing slots.

    Input structure:
    - bg = most common color. Exactly two distinct small prototype patterns appear as scaled sprite instances.
    - The three sprites occupy either one horizontal row of slots or one vertical column of slots, but their sizes are not initially sorted along that axis.

    Rule:
    - Recover the two prototype patterns from the smallest sprites and infer each sprite's prototype type and integer scale.
    - Determine whether the slots run left-to-right or top-to-bottom, then sort the sprites by scale from small to large.
    - Replace each sprite with the opposite prototype at the same scale, assigning the sorted swapped sprites back into the ordered slots.

    Key insight: The old per-sprite overlap alignment no longer decides destination; slots come from the global row/column layout, and scale order alone decides which swapped sprite goes into which slot.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=8)
    assert len(comps) >= 2, "grid must contain at least 2 sprite components"

    sprites: list[dict[str, object]] = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        sprites.append({"pts": pts, "bbox": (r0, c0, r1, c1), "arr": arr})

    areas = [int((s["bbox"][2] - s["bbox"][0] + 1) * (s["bbox"][3] - s["bbox"][1] + 1)) for s in sprites]
    min_area = min(areas)
    base = [s for s, a in zip(sprites, areas, strict=False) if a == min_area]
    uniq: dict[tuple[int, ...], np.ndarray] = {}
    for s in base:
        key = tuple(np.asarray(s["arr"]).ravel().tolist())
        if key not in uniq:
            uniq[key] = np.asarray(s["arr"]).copy()
    assert len(uniq) == 2, "smallest-area sprites must yield exactly 2 distinct prototype patterns"

    proto_keys = list(uniq.keys())[:2]
    protos = [uniq[proto_keys[0]], uniq[proto_keys[1]]]

    anc0 = _pick_anchor_bbox(protos[0], bg)
    a0r0, a0c0, a0r1, a0c1 = anc0
    base_anc_h = a0r1 - a0r0 + 1
    base_anc_w = a0c1 - a0c0 + 1

    centers = [_sprite_center_key(s) for s in sprites]
    row_range = max(rc[0] for rc in centers) - min(rc[0] for rc in centers)
    col_range = max(rc[1] for rc in centers) - min(rc[1] for rc in centers)
    horizontal = col_range >= row_range

    if horizontal:
        slots = sorted(sprites, key=lambda s: (int(s["bbox"][1]), int(s["bbox"][0])))
    else:
        slots = sorted(sprites, key=lambda s: (int(s["bbox"][0]), int(s["bbox"][1])))

    records: list[dict[str, object]] = []
    for slot_index, s in enumerate(slots):
        r0, c0, _r1, _c1 = s["bbox"]
        arr = np.asarray(s["arr"])
        ar0, ac0, ar1, ac1 = _pick_anchor_bbox(arr, bg)
        inst_anc_h = ar1 - ar0 + 1
        inst_anc_w = ac1 - ac0 + 1
        scale_h = max(1, inst_anc_h // max(1, base_anc_h))
        scale_w = max(1, inst_anc_w // max(1, base_anc_w))
        scale = int(max(scale_h, scale_w))
        proto_idx = _match_sprite(arr, protos, scale)
        records.append(
            {
                "slot_index": slot_index,
                "slot_rc": (int(r0), int(c0)),
                "scale": scale,
                "proto_idx": proto_idx,
            }
        )

    shapes = sorted(records, key=lambda rec: (int(rec["scale"]), int(rec["slot_index"])))
    out = np.full_like(g, bg)

    for dst, shape in zip(slots, shapes, strict=True):
        dst_r0, dst_c0, _dst_r1, _dst_c1 = dst["bbox"]
        repl = _scale_nn(protos[1 - int(shape["proto_idx"])], int(shape["scale"]))
        paste(out, repl, int(dst_r0), int(dst_c0), transparent=bg)

    return out


# --------------- generator (D3 variant) ---------------

# --- CFG: canvas ---
GRID_SIZE_MIN = 18
GRID_SIZE_MAX = 28
GRID_BORDER = 1
SLOT_GAP = 1
MAX_RETRIES = 300

# --- CFG: prototypes ---
PROTO_SHAPES = [(4, 3), (3, 4), (4, 4)]
BG_CHOICES = [0, 2, 3, 4, 5, 6, 7, 8, 9]
COLOR_SHARE_PROB = 0.65

# --- CFG: row-mask weights (anchor role vs variable role) ---
ANCHOR_ROW_WEIGHTS = [("center", 0.45), ("outer", 0.35), ("full", 0.20)]
VARIABLE_ROW_WEIGHTS = [("center", 0.68), ("full", 0.32)]
OUTER_FIXUP_PROB = 0.7

# --- CFG: slot layout ---
ORIENTATIONS = ["horizontal", "vertical"]
INSTANCES_TEMPLATE = [(0, 1), (1, 1), (None, 2)]  # last type filled randomly

_ROW_MASKS = {
    "center": {3: [0, 1, 0], 4: [0, 1, 1, 0]},
    "outer": {3: [1, 0, 1], 4: [1, 0, 0, 1]},
    "full": {3: [1, 1, 1], 4: [1, 1, 1, 1]},
}


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
    for r, (role, mask) in enumerate(zip(roles, masks, strict=True)):
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


def _assemble_scene(
    n: int,
    bg: int,
    protos: list[np.ndarray],
    slots: list[tuple[int, int]],
    assignments: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((n, n), bg, dtype=int)
    for (r0, c0), (typ, scale) in zip(slots, assignments, strict=True):
        patch = _scale(protos[typ], scale)
        paste(grid, patch, int(r0), int(c0), transparent=bg)
    return grid


def _ordered_output_assignments(assignments: list[tuple[int, int]]) -> list[tuple[int, int]]:
    sorted_shapes = sorted(
        list(enumerate(assignments)),
        key=lambda item: (int(item[1][1]), int(item[0])),
    )
    return [(1 - int(typ), int(scale)) for _idx, (typ, scale) in sorted_shapes]


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_CHOICES)
    proto_a, proto_b = _make_prototypes(bg)
    protos = [proto_a, proto_b]

    instances = [(0, 1), (1, 1), (random.randrange(2), 2)]
    orientation = random.choice(ORIENTATIONS)

    max_h = max(_scale(proto, scale).shape[0] for proto in protos for _typ, scale in instances)
    max_w = max(_scale(proto, scale).shape[1] for proto in protos for _typ, scale in instances)

    if orientation == "horizontal":
        need = max(2 * GRID_BORDER + max_h, 2 * GRID_BORDER + 3 * max_w + 2 * SLOT_GAP)
    else:
        need = max(2 * GRID_BORDER + max_w, 2 * GRID_BORDER + 3 * max_h + 2 * SLOT_GAP)
    if need > GRID_SIZE_MAX:
        return None
    n = random.randint(max(GRID_SIZE_MIN, need), GRID_SIZE_MAX)

    if orientation == "horizontal":
        row = random.randint(GRID_BORDER, n - GRID_BORDER - max_h)
        cols = _slot_offsets(n, 3, max_w, GRID_BORDER, SLOT_GAP)
        slots = [(int(row), int(c)) for c in cols]
    else:
        col = random.randint(GRID_BORDER, n - GRID_BORDER - max_w)
        rows = _slot_offsets(n, 3, max_h, GRID_BORDER, SLOT_GAP)
        slots = [(int(r), int(col)) for r in rows]

    perm = list(range(3))
    random.shuffle(perm)
    while perm == [0, 1, 2] or perm.index(2) == 2:
        random.shuffle(perm)
    input_assignments = [instances[i] for i in perm]
    output_assignments = _ordered_output_assignments(input_assignments)

    inp = _assemble_scene(n, bg, protos, slots, input_assignments)
    out = _assemble_scene(n, bg, protos, slots, output_assignments)

    try:
        solved = main(inp)
    except Exception:
        return None
    if np.array_equal(inp, out):
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid c7f57c3e D3 input")
